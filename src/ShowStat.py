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
    train_acc = [0.8878, 0.9295, 0.9216, 0.9449, 0.9537, 0.9559, 0.9619, 0.965, 0.9656, 0.971, 0.9723, 0.9726, 0.977, 0.977, 0.9804, 0.9778, 0.98, 0.9837, 0.9785, 0.9848, 0.9855, 0.987, 0.9884, 0.99, 0.9845, 0.9882, 0.9911, 0.9926, 0.9927, 0.9921, 0.9843, 0.9869, 0.9939, 0.9955, 0.9956, 0.9908, 0.9947, 0.994, 0.9978, 0.994, 0.994, 0.994, 0.994, 0.9946, 0.9957, 0.9965, 0.9958, 0.9963, 0.9926]
    train_loss = [0.3212, 0.2066, 0.2356, 0.1573, 0.1308, 0.1224, 0.1057, 0.1002, 0.0955, 0.0861, 0.0783, 0.0763, 0.0659, 0.0634, 0.0569, 0.0636, 0.0578, 0.049, 0.0613, 0.0422, 0.0418, 0.0368, 0.0326, 0.0278, 0.0462, 0.0347, 0.0246, 0.0215, 0.0206, 0.0212, 0.0515, 0.0395, 0.0178, 0.0139, 0.0128, 0.0257, 0.0167, 0.0187, 0.0068, 0.0187, 0.0187, 0.0187, 0.0187, 0.018, 0.0123, 0.0112, 0.012, 0.0117, 0.0241]
    val_acc = [0.9539, 0.9993, 0.9347, 0.9413, 0.9786, 0.997, 0.9898, 0.9756, 0.9578, 0.9947, 0.9845, 0.9881, 0.9064, 0.8873, 0.9802, 0.9812, 0.9769, 0.9862, 0.9156, 0.8701, 0.9944, 0.9951, 0.9769, 0.9384, 0.975, 0.943, 0.9684, 0.9911, 0.9934, 0.9677, 0.9881, 0.9865, 0.9097, 0.9575, 0.9736, 0.9759, 0.9581, 0.9878, 0.9548, 0.9878, 0.95, 0.988, 0.993, 0.998, 0.9964, 0.9868, 0.997, 0.9993, 0.9983]
    val_loss = [0.1643, 0.0031, 0.4075, 0.1723, 0.0977, 0.035, 0.0531, 0.0828, 0.1331, 0.0333, 0.0626, 0.0562, 0.3028, 0.3796, 0.0986, 0.0896, 0.1164, 0.0688, 0.2471, 0.4437, 0.0228, 0.0234, 0.09, 0.2509, 0.0811, 0.1856, 0.0903, 0.0455, 0.0332, 0.1153, 0.0595, 0.0632, 0.3865, 0.1923, 0.1273, 0.1055, 0.1926, 0.0554, 0.2017, 0.0554, 0.1926, 0.0632, 0.0234, 0.0066, 0.0134, 0.0337, 0.009, 0.0038, 0.0086]
    epoch = []
    e = 57
    while line:
        l = line.split("-")

        l2 = l[1].split(":")
        val_acc.append(float(l2[1]))
        l2 = l[2].split(":")
        train_acc.append(float(l2[1]))
        l2 = l[3].split(":")
        val_loss.append(float(l2[1]))
        l2 = l[4].split(":")
        l2 = l2[1].split(".h5")

        train_loss.append(float(l2[0]))

        epoch.append(e)
        e += 1
        line = fp.readline()
    print("train acc",train_acc)
    print("train loss",train_loss)
    print("val acc",val_acc)
    print("val loss",val_loss)
    showStat(train_acc,val_acc,train_loss,val_loss,10)

