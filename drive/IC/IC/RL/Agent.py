class AgentModel(nn.Module):
  def __init__(self, eff_core, n_classes, neighbor_size=(96, 96), compression_size=256, lstm_n_layers=2, lstm_hidden_size=256):
    self.neighbor_size = neighbor_size
    self.compression_size = compression_size
    self.lstm_hidden_size = lstm_hidden_size
    self.backbone = eff_core
    self.temporal_encoder = nn.ModuleList([
      nn.LSTMCell(compression_size * 2 if i == 0 else lstm_hidden_size, lstm_hidden_size) for i in range(lstm_n_layers)
    ])
    self.core_feature_compressor = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1536, self.compression_size),
        nn.BatchNorm1d(self.compression_size)
    )
    self.rel_feature_compressor = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1536, self.compression_size),
        nn.BatchNorm1d(self.compression_size)
    )
#     self.temporal_encoder = nn.LSTM(
#         input_size=self.compression_size * 2,
#         hidden_size=self.compression_size,
#         num_layers=2,
#         batch_first=True,
#     )
    self.action_head = nn.Sequential(
      nn.Linear(self.lstm_hidden_size, 64),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Linear(64, 8)
    )
    self.prediction_head = nn.Sequential(
        nn.Linear(self.lstm_hidden_size, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, n_classes)
    )
    self.history = []

  def clean(self):
    self.history = []

  def forward(self, image_grids, indices, hidden_state):
    h, c = hidden_state
    bsz = image_grids.shape[0]
    all_neighbors = []
    all_centers = []
    for bid in range(bsz):
      neighbors = []
      for i in range(-1, 2):
        neighbors.append([])
        for j in range(-1, 2):
          neighbors[-1].append(image_grids[bid][indices[bid][1]+i, indices[bid][0]+j])
        neighbors[-1] = np.stack(neighbors[-1])
      neighbors = np.stack(neighbors)
      neighbors = einops.rearrange(neighbors, 'N M h w C -> (N h) (M w) C')
      neighbors = cv2.resize(neighbors, dsize=(self.neighbor_size[0], self.neighbor_size[1]), interpolation=cv2.INTER_CUBIC)      
      all_neighbors.append(transform(neighbors))
      center = image_grids[bid][indices[bid][1], indices[bid][0]]
      all_centers.append(transform(center))
    all_neigbors = torch.stack(all_neighbors)
    all_centers = torch.stack(all_centers)
    neighbors_feature = model.forward_features(all_neighbors)
    self_feature = torch.mean(model.forward_features(all_centers), dim=(-1, -2)).unsuqeeze(1)
    neighbors_feature = einops.rearrange(neighbors_feature, 'N F h w -> N (h w) F')
    neighbors_feature = torch.cat([neighbors_feature[:, :4], neighbors_feature[:, 5:]], dim=1)
    rel = attention(self_feature, neighbors_feature, neighbors_feature).squeeze(1)
    core = self_feature.squeeze(1)
    complete = torch.cat([self.core_feature_compressor(core), self.rel_feature_compressor(rel)], dim=-1)
    #TODO
    # feed features to LSTM
    output_hidden_state = ([], [])
    for i in range(len(self.temporal_encoder)):
      x_in, h_in, c_in = complete if i == 0 else h_out, h[i], c[i]
      h_out, c_out = self.temporal_encoder[i](x_in, (h_in, c_in))
      output_hidden_state[0].append(h_out)
      output_hidden_state[1].append(c_out)
      
    self.history.append((complete, output_hidden_state[0][-1]))
      
    return self.action_head(output_hidden_state[0][-1]), self.prediction_head(output_hidden_state[0][-1]), output_hidden_state
