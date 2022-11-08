# gradient descent to find paramters of line
import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path



# create parameters to find output values
weight = 0.7
bias = 0.3

# create data
start = 0
end = 1 
step = 0.02

X = torch.arange(start, end, step)
X = X.unsqueeze(dim = 1)

Y = weight * X + bias

# print the data
print(X[:10], Y[:10])

# create train data and test data
# 80 % of data to train and rest to test
train_split = int(0.8 * len(X))
X_train, Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split:], Y[train_split:]

#print(len(X_train), len(Y_train), len(X_test), len(Y_test))

# function to plot data
def plot_data(train_data = X_train, train_labels = Y_train, test_data = X_test, test_labels = Y_test, output = None):
    plt.figure(figsize = (10,7))

    # blue color test data
    plt.scatter(train_data, train_labels, c = "b", s = 4, label = "training data")

    #test data
    plt.scatter(test_data, test_labels, c = "g", s = 4, label = "test data")

    if output is not None:
        plt.scatter(test_data, output, c = "r", s = 4, label = "output")

    plt.legend(prop={"size": 14})

    plt.show()


#plot_data()

# build model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # create paramters and start with random values
        self.weights = nn.Parameter(torch.randn(1, requires_grad = True, dtype = torch.float))

        self.bias = nn.Parameter(torch.randn(1, requires_grad = True, dtype = torch.float))

    # inbuild method overriding
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias



# set the seed to make random values same every time we run the model
torch.manual_seed(42)

# instantiace the model
model_0 = LinearRegression()

print(model_0.state_dict())

# try to pridict without training on test data
# with torch.inference_mode():
#     y_preds = model_0(X_test)

# plot_data(output=y_preds)

# Create loss function
loss_fn = nn.L1Loss()
# lr = change in paramter by optimizer
optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.01)


# training loop
epochs = 250

epoch_count = []
train_loss_values = []
test_loss_values = []

for epoch in range(epochs):

    # put model in training mode
    model_0.train()

    # forward pass 
    y_preds = model_0(X_train)

    # check loss by checking again actual data
    loss = loss_fn(y_preds, Y_train)

    # zero gradient of optimizer
    optimizer.zero_grad()

    # loss backwards
    loss.backward()

    optimizer.step()

    # testing mode
    model_0.eval()

    with torch.inference_mode():
        test_pred = model_0(X_test)

        # predictions come in torch.float type
        test_loss = loss_fn(test_pred, Y_test.type(torch.float))

        # print output every 10th time
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")




# plt.plot(epoch_count, train_loss_values, label="Train loss")
# plt.plot(epoch_count, test_loss_values, label="Test loss")
# plt.title("Training and test loss curves")
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.legend();
# plt.show()


print(model_0.state_dict())

model_0.eval()

with torch.inference_mode():
    y_pred = model_0(X_test)

plot_data(output=y_pred)

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
            f=MODEL_SAVE_PATH)



# load the saved model
loaded_model_0 = LinearRegression()
# load the parameters 
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# 1. Put the loaded model into evaluation mode
loaded_model_0.eval()

# 2. Use the inference mode context manager to make predictions
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)

# compare with previous model
#print(loaded_model_preds)

plot_data(output=loaded_model_preds)








