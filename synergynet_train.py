import os
import os.path as osp
from pathlib import Path
import numpy as np
import argparse
import time
import logging
import datetime

import tensorflow as tf

from utilstf.ddfa import DDFADataset
from utilstf.io import mkdir
from utilstf.ddfa import str2bool
from synergynet_tf import SynergyNet as SynergyNet
from synergynet_loss_def import ParamAcc, TrainLoss, LmkLoss, LmkAcc
from synergynet_image_plot_callback import ImagePlotCallback
from training_time_callback import TrainTimeCallback


# global args (configuration)
args = None # define the static training setting, which wouldn't and shouldn't be changed over the whole experiements.

def parse_args():
    parser = argparse.ArgumentParser(description='3DMM Fitting')
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--start-epoch', default=1, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('-vb', '--val-batch-size', default=32, type=int)
    parser.add_argument('--base-lr', '--learning-rate', default=0.08, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
    parser.add_argument('--print-freq', '-p', default=20, type=int)
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--resume_pose', default='', type=str, metavar='PATH')
    parser.add_argument('--devices-id', default='0', type=str)
    parser.add_argument('--filelists-train',default='./3dmm_data/train_aug_120x120.list.train', type=str)
    parser.add_argument('--root', default='./train_aug_120x120')
    parser.add_argument('--snapshot', default='', type=str)
    parser.add_argument('--log-file', default='output.log', type=str)
    parser.add_argument('--log-mode', default='w', type=str)
    parser.add_argument('--arch', default='mobilenet_v2', type=str, help="Please choose [mobilenet_v2, mobilenet_1, resnet50, resnet101, or ghostnet]")
    parser.add_argument('--milestones', default='15,25,30', type=str)
    parser.add_argument('--task', default='all', type=str)
    parser.add_argument('--test_initial', default='True', type=str2bool)
    parser.add_argument('--warmup', default=5, type=int)
    parser.add_argument('--param-fp-train',default='./3dmm_data/param_all_norm_v201.pkl',type=str)
    parser.add_argument('--img_size', default=120, type=int)
    parser.add_argument('--save_val_freq', default=10, type=int)

    global args
    args = parser.parse_args()

    # some other operations
    args.devices_id = [int(d) for d in args.devices_id.split(',')]
    args.milestones = [int(m) for m in args.milestones.split(',')]

    snapshot_dir = osp.split(args.snapshot)[0]
    mkdir(snapshot_dir)

    return args

def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)

def save_checkpoint(model, filename='checkpoint.pth.tar'):
    checkpoint = tf.train.Checkpoint(model)
    save_path = checkpoint.save(filename)
    logging.info(f'Save checkpoint to {filename}')
    
    return save_path

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#@tf.function
# def train_step(model, x_batch_train, y_batch_train, optimizer, loss, train_acc_metric):
#     with tf.GradientTape() as tape:
#         pred = model(x_batch_train, training=True)
#         loss_value = loss(y_batch_train, pred)            
#     grads = tape.gradient(loss_value, model.trainable_weights)
#     optimizer.apply_gradients(zip(grads, model.trainable_weights))
#     # Update training metric.
#     train_acc_metric.update_state(y_batch_train, pred)
    
#     return loss_value

# def train(model, loss, train_acc_metric, val_acc_metric,
#           optimizer, train_data, val_data, args, callback_list=[]):
#     """Custom training

#     Args:
#         model (face align model): image to parameters
#         loss: loss function
#         optimizer: optimizer
#         args (training params): some training configurations
#     """    
#     callbacks = tf.keras.callbacks.CallbackList(callback_list, add_history=True, add_progbar=False, model=model)
    
#     total_batch = train_data.cardinality()
#     callbacks.on_train_begin()
#     for epoch in range(args.epochs):
#     #for epoch in range(1):
        
#         callbacks.on_epoch_begin(epoch)
#         print("\nStart of epoch %d" % (epoch,))
#         start_time = time.time()
        
#         loss_value = np.inf
#         for step, (x_batch_train, y_batch_train) in enumerate(train_data):
#             callbacks.on_train_batch_begin(step)
            
#             loss_value = train_step(model, x_batch_train, y_batch_train, optimizer, loss, train_acc_metric)
            
#             callbacks.on_train_batch_end(step, logs=loss_value)
#             if step % 200 == 0:
#                 print(
#                     "Training loss (for one batch) at step %d: %.4f"
#                     % (step, float(loss_value)) 
#                 )
#                 print("Seen so far: %d/%d samples" % ((step+1)*args.batch_size, total_batch*args.batch_size))
                
                
                 
#         # Display metrics at the end of each epoch.
#         train_acc = train_acc_metric.result()
#         print("Training accuracy over epoch: %.4f" % (float(train_acc),))
        
        
#         train_acc_metric.reset_states()
        
#         # Run a validation loop at the end of epoch
#         for x_batch_val, y_batch_val in val_data:
#             param_val = model(x_batch_val, training=False)
#             # update val metrics
#             val_acc_metric.update_state(y_batch_val, param_val)
#         val_acc = val_acc_metric.result()
#         val_acc_metric.reset_states()
        
#         callbacks.on_epoch_end(epoch, logs={"loss":loss_value, "acc":train_acc, "val_loss":val_acc, "val_acc":val_acc})
#         print("Validation acc: %.4f" % (float(val_acc),))
#         print("Time taken: %.2fs" % (time.time()-start_time))
        
#     callbacks.on_train_end()
        
         
def main():
    """ Main function for the training process"""
    parse_args()
    
    # logging setup
    logging.basicConfig(
        format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode=args.log_mode),
            logging.StreamHandler()
        ]
    )
    
    print_args(args)
    
    
    # prepare dataset
    train_dataset, val_dataset, test_dataset = DDFADataset(
        root=args.root,
        filelists=args.filelists_train,
        param_fp=args.param_fp_train,
        batch_size= args.batch_size,
        gt_transform=True,
        transform=[]
    )
    print("Number of training batches: ", train_dataset.cardinality().numpy())
    
    model = SynergyNet(args)
    
    # Resume
    resume = False
    if resume==True:
        resume_model = os.path.join('./ckpts', 'cp-0038.ckpt')
        model.load_weights(resume_model)
    
    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.base_lr,
        decay_steps=train_dataset.cardinality().numpy(),
        decay_rate=0.96)
    
    # Optimizer
    optimizer_sgd = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    
    # Loss
    train_loss = TrainLoss()
    lmk_loss = LmkLoss()
    #train_loss = tf.keras.losses.MeanSquaredError()
    train_acc = ParamAcc()
    val_acc = ParamAcc()
    lmk_acc = LmkAcc()

    # Compile model
    model.compile(
        optimizer = optimizer_sgd,
        loss = {"pred_param": train_loss, "pred_lmk": lmk_loss, "refined_lmk": lmk_loss, "refined_param": train_loss},
        loss_weights = [0.02, 0.05, 0.05, 0.02],
        metrics = {"pred_param": [], "pred_lmk": lmk_acc, "refined_lmk": [], "refined_param": []},
    )
    model.summary()
       
    # callbacks
    ckpt_folder = "./ckpts_mod"
    checkpoint_path = os.path.join(ckpt_folder, "cp-{epoch:04d}.ckpt")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     save_best_only=True)
    
    log_dir = "tensorboard_builtin/" + "new_mlp_for_back" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
    
    file_writer = tf.summary.create_file_writer(log_dir)
    
    train_img, train_param = next(iter(train_dataset))
    val_img, val_param = next(iter(val_dataset))
    image_plot_callback = ImagePlotCallback(train_img, train_param, val_img, file_writer)
    
    training_time_callback = TrainTimeCallback(file_writer)
    

    # Start training
    model.fit(train_dataset, 
              epochs=80,
              steps_per_epoch=None,
              callbacks=[checkpoint_callback, tensorboard_callback, image_plot_callback, training_time_callback],
              validation_data=val_dataset)    
    

    
if __name__ == '__main__':
    main()    