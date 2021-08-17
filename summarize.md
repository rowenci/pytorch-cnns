loss一直保持2.3

说明的网络结构有问题，神经网络的参数没有更新，也就是没有在学习，你更改网络机构或者其他的参数试一下

sample training set 做sanity check

我认为是超参数设置不对的可能性更大

loss下降的同时accuracy下降表示进入overfitting的领域，可以想象成你的模型在训练中依然进步，但是在validation set上测出的accuracy已经开始下降了。

接下来，accuracy反升，这表示你的模型中的regularization开始起作用，在成功遏制overfitting，于是你的accuracy开始回升，但是作为代价你的loss会退步，因为你的模型在阻止自己过度拟合训练数据。

简单来说，双降表示你的模型在训练数据中一头扎得太深，双升表示你的模型意识到这个问题，开始回过头来补救。

在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。如果设置初始化，则每次初始化都是固定的。
if args.seed is not None:
　　random.seed(args.seed) #
　　torch.manual_seed(args.seed)  #为CPU设置种子用于生成随机数，以使得结果是确定的
　　 torch.cuda.manual_seed(args.seed) #为当前GPU设置随机种子；
　　  cudnn.deterministic = True
