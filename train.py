import torch
import pandas as pd

def train(model, device, train_data, val_data, epochs, optimizer, focal_loss, save_dir):

    model.to(device)
    best_val_loss = float('inf')

    train_loss_arr = []
    val_loss_arr = []
    best_epoch = 0
    df = pd.DataFrame(columns=["train_loss","val_loss"])

    for epoch in range(epochs):
        #Training Loop
        model.train()
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_data):
            y_truth = y["masks_faces"]
            x, y_truth = x.to(device), y_truth.to(device)
            optimizer.zero_grad()
            y_pred  = model(x)
            loss = focal_loss(y_pred, y_truth, 0.25, 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_data)

        #Validation loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x_val, y_val) in enumerate(val_data):
                y_truth = y_val["masks_faces"]
                x_val, y_truth = x_val.to(device), y_truth.to(device)
                y_pred = model(x_val)
                loss = focal_loss(y_pred, y_truth, 0.25, 2)
                total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_data)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_dir)
                best_epoch = epoch

        #save losses to df
        df.loc[len(df.index)] = [avg_train_loss,avg_val_loss]
        df.to_csv(save_dir[:-3]+".csv") 
        #Print Train and Val losses to terminal
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


