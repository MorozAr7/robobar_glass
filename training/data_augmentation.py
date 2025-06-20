import albumentations as A

data_aug = A.Compose([
    A.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 1.5), saturation=(1, 3), hue=(0, 0), always_apply=False, p=0.9),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
    A.PixelDropout(dropout_prob=0.01, per_channel=False, drop_value=0, mask_drop_value=None, p=0.5, always_apply=None),
    
    A.ToGray(p=0.1, num_output_channels=3),
    A.ChannelShuffle(p=0.1),
    
    A.OneOf([
        A.GaussNoise(var_limit=(25.0, 40.0), mean=0, per_channel=True, always_apply=False, p=1),
        A.MultiplicativeNoise(multiplier=(0.925, 1.075), per_channel=True, elementwise=True, always_apply=False, p=1),
        A.ISONoise(color_shift=(0.03, 0.06), intensity=(0.1, 0.4), always_apply=False, p=1)

    ], p=0.5),

    # A.OneOf([
    #     A.Blur(p=1, blur_limit=3),
    #     A.GaussianBlur(p=1, blur_limit=(1, 3), sigma_limit=0),
    # ], p=0.25),

], p=1.0)
