from typing import Optional
import torch
import torch.nn as nn
from d3nav.model.d3nav import D3Nav

class D3NavIDD(D3Nav):

    def __init__(
        self,
        temporal_context: int = 2,        # num frames input
        num_unfrozen_layers: int = 3,     # num GPT layers unfrozen
    ):
        super(D3NavIDD, self).__init__(
            load_comma=True,
            temporal_context=temporal_context,
        )

        # Freeze the entire model initially
        self.freeze_vqvae()
        self.freeze_gpt()
        
        # Then unfreeze only the specified number of GPT layers
        self.unfreeze_last_n_layers(num_unfrozen_layers)
    
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
        # assert T == self.T
        x = x.reshape(B * T, C, H, W)
        # x: (B, T, 3, 128, 256)
        # y: (B, 1, 3, 128, 256)
        

        z, z_history_feats = self.encoder(x, return_feats=True)
        z_history_feats = z_history_feats.reshape(B, T, 256, 8, 16)
        # z: (B*T, 128)
        # z_history_feats: (B, T, 256, 8, 16)

        z = z.to(dtype=torch.int32)
        z = z.reshape(B, T, -1)
        # z: (B, T, 128)

        # Create BOS tokens
        bos_tokens = torch.full(
            (B, T, 1),
            self.config_gpt.bos_token,
            dtype=z.dtype,
            device=z.device,
        )

        # Concatenate BOS tokens with z
        z = torch.cat([bos_tokens, z], dim=2)
        # z: (B, T, 129)

        zp_l = []
        zp_l_probs = []
        for index in range(B):
            zp_i, zp_i_probs = self.generate(
                z[index].reshape(T * self.config_gpt.tokens_per_frame),
                self.config_gpt.tokens_per_frame,
            )
            zp_l.append(zp_i)
            zp_l_probs.append(zp_i_probs)

        zp: torch.Tensor = torch.cat(zp_l)
        # zp: [B*129]
        zp = zp.reshape(B, self.config_gpt.tokens_per_frame)
        # zp: [B, 129]

        zp_probs: torch.Tensor = torch.cat(zp_l_probs)
        # zp_probs: (B, 1, 128, 1025)

        zp = zp[:, 1:]
        # zp: [B, 128]
        zp = zp.to(dtype=torch.int64)

        xp, z_feat = self.decoder(zp, return_feats=True)
        xp = xp.reshape(B, 1, C, H, W)


        if train_mode:
            y = y.reshape(B * 1, C, H, W)
            # y: (B*1, 3, 128, 256)

            yz = self.encoder(y)
            # yz: (B*1, 128)

            ygt = self.decoder(yz)
            ygt = ygt.view(B, 1, C, H, W)

            yz = yz.reshape(B*1*128)

            zp_probs = zp_probs.reshape(B*1*128, 1025)

            loss = torch.nn.functional.cross_entropy(
                zp_probs,
                yz,
            )

            return xp, ygt, loss

        return xp
    
    def generate(
        self, prompt: torch.Tensor, max_new_tokens: int
    ) -> torch.Tensor:
        t = prompt.size(0)
        T_new = t + max_new_tokens
        max_seq_length = self.model.config.block_size
        device, dtype = prompt.device, prompt.dtype
        # with torch.device(device):
        #     self.model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

        seq = torch.empty(T_new, dtype=dtype, device=device).clone()
        seq[:t] = prompt
        input_pos = torch.arange(0, t, device=device)
        next_token = self.model.prefill(prompt.view(1, -1), input_pos).clone()
        seq[t] = next_token
        input_pos = torch.tensor([t], device=device, dtype=torch.int)
        generated_tokens, generated_tokens_probs = self.model.decode_n_tokens(
            next_token.view(1, -1), input_pos, max_new_tokens - 1
        )
        generated_tokens_probs = torch.cat(generated_tokens_probs)
        generated_tokens_probs = generated_tokens_probs.view(1, 1, max_new_tokens-1, -1)
        seq[t + 1 :] = torch.cat(generated_tokens)
        return seq[t:], generated_tokens_probs

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
    B, T, C = 1, 2, 3
    x = torch.zeros((B, T, C, H, W), requires_grad=True)
    
    # Put the images into x
    x.data[0, 0] = img_1
    x.data[0, 1] = img_2
    
    # Expected Output Image: 1xRGB (0-255)    
    y = torch.zeros((B, 1, C, H, W), requires_grad=True)
    
    # Put the third image into y
    y.data[0, 0] = img_3
    
    model = D3NavIDD()
    model.unfreeze_last_n_layers(num_layers=1)

    print("x", x.shape, x.dtype, (x.min(), x.max()))

    # Predicted Future Image: 1xRGB (0-255)
    yp, ygt, loss = model(
        x=x,
        y=y,
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
