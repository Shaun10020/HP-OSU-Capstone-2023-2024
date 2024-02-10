# import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models.segmentation as segmentation

# using crossEntropy Loss and Adam optimizer for demonstration
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10  # Define the number of epochs

for epoch in range(num_epochs):
  # ... import the model before set it to train
  # ... like 'model = CustomDeepLabV3()' for DeepLabV3 model
  model.train()
    
  for module in model.modules():
      if isinstance(module, nn.BatchNorm2d):
          module.eval()
  running_loss = 0.0

  for inputs, char1, char2, char3 in train_loader:
      # Move data to the appropriate device (CPU/GPU)
      stacked_tensor = torch.stack([char1, char2, char3], dim=1)
      masks = torch.squeeze(stacked_tensor, 2)
      inputs, masks = inputs.to(device), masks.to(device)

      # Zero the parameter gradients
      optimizer.zero_grad()

      # Forward pass
      # make sure the output has gone through a sigmoid pass -- 'torch.sigmoid(output)'
      # in the forward function
      outputs = model(inputs.float())

      # Compute loss
      loss = criterion(outputs, masks)

      # Backward pass and optimize
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

  epoch_loss = running_loss / len(train_loader)
  print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Add validation and saving checkpoints if needed
