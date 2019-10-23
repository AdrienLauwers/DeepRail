import matplotlib.pyplot as plt
def showStat(acc,val_acc,loss,val_loss,initial_epochs):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.5, 1])
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

with open('stat.txt') as fp:
    line = fp.readline()
    cnt = 1
    val_acc = []
    val_loss = []
    train_acc = []
    train_loss = []
    epoch = []
    e = 8
    while line:
        if cnt%2 == 0:
            l = line.split()
            train_loss.append(float(l[7]))
            train_acc.append(float(l[10]))
            val_loss.append(float(l[13]))
            val_acc.append(float(l[16]))
            epoch.append(e)
            e += 1
        line = fp.readline()
        cnt += 1
    print("train acc",train_acc)
    print("train loss",train_loss)
    print("val acc",val_acc)
    print("val loss",val_loss)
    showStat(train_acc,val_acc,train_loss,val_loss,10)

