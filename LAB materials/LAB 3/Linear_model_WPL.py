x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0 # a random guess : random value 

# our model forward pass
def forward(x):
    return x * w

#loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# compute gradient
def gradient(x, y): #d_loss.d_w
    return 2 * x * ( x * 2 - y)

# Before training 
print("prediction (before)")