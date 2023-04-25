def images_to_grid(images_list, grid_size=(64, 64)):
  image_grids = []
  for i, image in enumerate(images_list):
    image = np.asarray(image)
    h, w, _ = image.shape
    W = ((w + grid_size[0] - 1) // grid_size[0]) * grid_size[0] 
    H = ((h + grid_size[1] - 1) // grid_size[1]) * grid_size[1]
    image = np.pad(image, [((H-h)//2, (H-h)-(H-h)//2), ((W-w)//2, (W-w)-(W-w)//2), (0, 0)], mode='linear_ramp')
    image_grids.append(einops.rearrange(image, '(N h) (M w) C -> N M h w C', h=grid_size[1], w=grid_size[0]))
  return np.stack(image_grids)

transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
])


core_feature_compressor = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1536, 256),
    nn.BatchNorm1d(256)
)
rel_feature_compressor = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1536, 256),
    nn.BatchNorm1d(256)
)
core_feature_compressor.eval()
rel_feature_compressor.eval()