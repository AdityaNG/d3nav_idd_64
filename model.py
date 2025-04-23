from typing import Optional
import torch
import torch.nn as nn
from d3nav.model.d3nav import D3Nav, GPTConfig, GPT, DEFAULT_DATATYPE

class D3NavIDD(D3Nav):

    def __init__(
        self,
        temporal_context: int = 2,        # num frames input
        num_layers: int = 24,             # num GPT layers unfrozen
        use_comma_gpt: bool = True,
        attention_dropout_p: float = 0.1,
    ):
        super(D3NavIDD, self).__init__(
            load_comma=True,
            temporal_context=temporal_context,
            attention_dropout_p=attention_dropout_p,
        )

        if not use_comma_gpt:
            # Fresh Backbone, overwriting the previous backbone
            self.config_gpt = GPTConfig(
                n_layer=num_layers,
            )
            model = GPT(self.config_gpt)

            self.model = model.to(dtype=DEFAULT_DATATYPE)
        else:
            assert num_layers == 24, "Comma GPT is 24 layers"

        # Freeze the entire model initially
        self.freeze_vqvae()
        self.freeze_gpt(requires_grad=True)
    
    def quantize(self, x: torch.Tensor):
        """
            Quantizes an input image and returns the quantized features
            along with the decoded image.

            x -> (B, T, 3, 128, 256)

            z -> (B, T, 256)
            z_feats -> (B, T, 256, 8, 16)
            xp -> (B, T, 3, 128, 256)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        z, z_feats = self.encoder(x, return_feats=True)
        z_feats = z_feats.reshape(B, T, 256, 8, 16)

        xp = self.decoder(z)
        xp = xp.view(B, T, C, H, W)
        return z, z_feats, xp
    
    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass that processes input frames through GPT and decodes the output
        
        Args:
            x: Input tensor of shape (B, T, 3, 128, 256)
            y: Expected Output tensor of shape (B, T, 3, 128, 256)
            
        Returns:
            Decoded output image of the same shape
        """
        train_mode = y is not None
        
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        
        z, z_history_feats = self.encoder(x, return_feats=True)
        z_history_feats = z_history_feats.reshape(B, T, 256, 8, 16)
        
        z = z.to(dtype=torch.int32)
        z = z.reshape(B, T, -1)
        
        # Create BOS tokens
        bos_tokens = torch.full(
            (B, T, 1),
            self.config_gpt.bos_token,
            dtype=z.dtype,
            device=z.device,
        )
        
        # Concatenate BOS tokens with z
        z = torch.cat([bos_tokens, z], dim=2)  # (B, T, 129)
        
        # Reshape for processing
        z_flat = z.reshape(B, T * self.config_gpt.tokens_per_frame)
        
        if train_mode:
            # TEACHER FORCING: Use ground truth tokens for training
            # Process ground truth for loss calculation
            y = y.reshape(B * 1, C, H, W)
            yz = self.encoder(y)  # (B*1, 128)
            ygt = self.decoder(yz)
            ygt = ygt.view(B, 1, C, H, W)
            
            # Get loss using teacher forcing
            xp, loss = self._teacher_forcing_forward(z_flat, yz)
            
            return xp.reshape(B, 1, C, H, W), ygt, loss
        else:
            # For inference, use autoregressive generation
            zp_batch, _ = self.batch_generate(
                z_flat, 
                self.config_gpt.tokens_per_frame
            )
            
            # Remove BOS token
            zp = zp_batch[:, 1:]
            zp = zp.to(dtype=torch.int64)
            zp = torch.clamp(zp, min=0, max=1023)
            
            # Decode the predicted tokens
            xp, _ = self.decoder(zp, return_feats=True)
            return xp.reshape(B, 1, C, H, W)
    
    def _teacher_forcing_forward(self, context_tokens, target_tokens):
        """
        Perform teacher forcing forward pass for training
        
        Args:
            context_tokens: Context tokens from input frames (B, T*tokens_per_frame)
            target_tokens: Target tokens from ground truth frame (B*1, tokens_per_frame)
            
        Returns:
            Decoded output image and loss
        """
        B = context_tokens.size(0)
        device = context_tokens.device
        dtype = context_tokens.dtype
        
        # Reshape target tokens to (B, tokens_per_frame)
        target_tokens = target_tokens.reshape(B, -1)
        
        # Add BOS token to the target sequence for prediction
        target_with_bos = torch.cat([
            torch.full((B, 1), self.config_gpt.bos_token, dtype=target_tokens.dtype, device=device),
            target_tokens
        ], dim=1)
        
        # Create position indices for the context and target tokens
        context_len = context_tokens.size(1)
        target_len = target_with_bos.size(1)
        
        # For teacher forcing, we feed all tokens at once but create attention masks
        # that ensure each prediction only sees previous tokens
        input_tokens = torch.cat([context_tokens, target_with_bos[:, :-1]], dim=1)
        
        # Position indices for the input
        input_pos = torch.arange(0, context_len + target_len - 1, device=device).unsqueeze(0).repeat(B, 1)
        
        # Get predictions for all tokens at once
        logits = self.model(input_tokens, input_pos)
        
        # We only care about predictions for the target sequence
        target_logits = logits[:, context_len:, :]  # (B, target_len-1, vocab_size)
        
        # Flatten the logits and targets for cross-entropy loss
        flat_logits = target_logits.reshape(-1, target_logits.size(-1))
        flat_targets = target_tokens.reshape(-1)
        
        # Calculate cross-entropy loss
        loss = torch.nn.functional.cross_entropy(flat_logits, flat_targets)
        
        # For returning the predicted image, get the most likely token at each position
        pred_tokens = torch.argmax(target_logits, dim=-1)
        
        # Decode the predicted tokens
        pred_tokens = pred_tokens.to(dtype=torch.int64)
        pred_tokens = torch.clamp(pred_tokens, min=0, max=1023)
        
        xp, _ = self.decoder(pred_tokens, return_feats=True)
        
        return xp, loss
    
    def batch_generate(self, prompt: torch.Tensor, max_new_tokens: int):
        """
        Generate tokens for a batch of prompts, processing all batch items in parallel
        but still generating tokens auto-regressively.
        
        Args:
            prompt: Tensor of shape (B, seq_len) containing the prompt tokens
            max_new_tokens: Number of new tokens to generate
            
        Returns:
            Generated tokens and their probabilities
        """
        B = prompt.size(0)
        t = prompt.size(1)
        device, dtype = prompt.device, prompt.dtype
        
        # Set up storage for results
        generated_tokens = torch.empty((B, max_new_tokens), dtype=dtype, device=device)
        all_probs = []
        
        # First token generation - process all batch items in parallel
        input_pos = torch.arange(0, t, device=device).unsqueeze(0).repeat(B, 1)
        logits = self.model(prompt, input_pos)  # (B, seq_len, vocab_size)
        
        # Get probabilities for the last position
        probs = torch.nn.functional.softmax(logits[:, -1], dim=-1)  # (B, vocab_size)
        
        # Sample from the probability distribution
        next_tokens = torch.multinomial(probs, 1).squeeze(-1)  # (B)
        generated_tokens[:, 0] = next_tokens
        all_probs.append(probs)
        
        # Current context - just the sampled token for each batch item
        current_tokens = next_tokens.unsqueeze(-1)  # (B, 1)
        
        # Generate remaining tokens auto-regressively
        for i in range(1, max_new_tokens):
            # The position is the same for all batch items
            curr_pos = torch.full((B,), t + i - 1, device=device, dtype=torch.long)
            
            # Process all batch items in parallel, but only one new position at a time
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, 
                enable_mem_efficient=False, 
                enable_math=True
            ):
                logits = self.model(current_tokens, curr_pos.unsqueeze(-1))  # (B, 1, vocab_size)
            
            # Get probabilities
            probs = torch.nn.functional.softmax(logits[:, -1], dim=-1)  # (B, vocab_size)
            all_probs.append(probs)
            
            # Sample next tokens
            next_tokens = torch.multinomial(probs, 1).squeeze(-1)  # (B)
            generated_tokens[:, i] = next_tokens
            
            # Update current tokens
            current_tokens = next_tokens.unsqueeze(-1)  # (B, 1)
        
        # Stack all probability tensors
        all_probs = torch.stack(all_probs, dim=1)  # (B, max_new_tokens, vocab_size)
        all_probs = all_probs.unsqueeze(1)  # (B, 1, max_new_tokens, vocab_size)
        
        return generated_tokens, all_probs

if __name__ == "__main__":
    import os
    import torch
    import numpy as np
    import cv2

    torch.autograd.set_detect_anomaly(True)

    dataset_base = "/media/NG/datasets/idd_mini/idd_temporal_train4/029462_leftImg8bit"
    img_1 = cv2.imread(f"{dataset_base}/0003399.jpeg")
    img_2 = cv2.imread(f"{dataset_base}/0003400.jpeg")
    img_3 = cv2.imread(f"{dataset_base}/0003401.jpeg")

    # Convert images to PyTorch tensors
    H, W = 128, 256
    
    # Resize images
    img_1 = cv2.resize(img_1, (W, H))
    img_2 = cv2.resize(img_2, (W, H))
    img_3 = cv2.resize(img_3, (W, H))
    
    # Convert BGR to RGB
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
    img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] and convert to tensor
    img_1 = torch.tensor(img_1.transpose(2, 0, 1)).float()
    img_2 = torch.tensor(img_2.transpose(2, 0, 1)).float()
    img_3 = torch.tensor(img_3.transpose(2, 0, 1)).float()
    
    # Input Images: 2xRGB (0-255)
    B, T, C = 2, 2, 3
    x = torch.zeros((B, T, C, H, W), requires_grad=True)
    
    # Put the images into x
    x.data[0, 0] = img_1
    x.data[0, 1] = img_2
    
    # Expected Output Image: 1xRGB (0-255)    
    y = torch.zeros((B, 1, C, H, W), requires_grad=True)
    
    # Put the third image into y
    y.data[0, 0] = img_3
    
    model = D3NavIDD(
        use_comma_gpt=True,
        num_layers=24,
    )
    model = model.cuda()

    print("x", x.shape, x.dtype, (x.min(), x.max()))

    # Predicted Future Image: 1xRGB (0-255)
    yp, ygt, loss = model(
        x=x.cuda(),
        y=y.cuda(),
    )

    print('yp', yp.shape)

    # Test gradient flow
    loss.backward()

    print("x.grad is None:", x.grad is None)
    print("yp shape:", yp.shape)

    # Save the predicted image
    # First detach from computation graph and move to CPU if needed
    # pred_img = yp.detach().cpu()[0, 0]  # Get the first image from batch
    pred_img = ygt.detach().cpu()[0, 0]  # Ground truth encoded then decoded
    
    # Convert from [0,1] to [0,255] and from CHW to HWC format
    pred_img = pred_img.permute(1, 2, 0).numpy()  # Change to HWC format
    pred_img = np.clip(pred_img, 0, 255).astype(np.uint8)  # Scale to [0,255]
    
    # Convert RGB to BGR for OpenCV
    pred_img_bgr = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
    
    # Create output directory if it doesn't exist
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the predicted image
    cv2.imwrite(f"{output_dir}/predicted_0003401.jpg", pred_img_bgr)
    print(f"Predicted image saved to {output_dir}/predicted_0003401.jpg")
    
    # Also save the input and ground truth for comparison
    img1_bgr = cv2.cvtColor((x[0, 0].detach().cpu().permute(1, 2, 0).numpy()).astype(np.uint8), cv2.COLOR_RGB2BGR)
    img2_bgr = cv2.cvtColor((x[0, 1].detach().cpu().permute(1, 2, 0).numpy()).astype(np.uint8), cv2.COLOR_RGB2BGR)
    gt_bgr = cv2.cvtColor((y[0, 0].detach().cpu().permute(1, 2, 0).numpy()).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(f"{output_dir}/input_0003399.jpg", img1_bgr)
    cv2.imwrite(f"{output_dir}/input_0003400.jpg", img2_bgr)
    cv2.imwrite(f"{output_dir}/ground_truth_0003401.jpg", gt_bgr)
    print(f"Input and ground truth images saved to {output_dir}/")
