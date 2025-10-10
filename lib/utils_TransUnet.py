import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def calculate_iou(y_true, y_pred):
    """
    计算二值分割的IoU
    :param y_true: 真实掩膜（二值数组）
    :param y_pred: 预测掩膜（二值数组）
    :return: IoU值
    """
    # 展平数组
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    # 计算交集（TP）
    intersection = np.sum(y_true_f * y_pred_f)
    
    # 计算并集 = TP + FP + FN
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    
    # 避免除零错误
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou

def mean_iou(y_true, y_pred, num_classes):
    """
    计算多类别分割的Mean IoU
    :param y_true: 真实掩膜（整型数组，值∈[0, num_classes-1]）
    :param y_pred: 预测掩膜（整型数组）
    :param num_classes: 类别数
    :return: Mean IoU
    """
    ious = []
    for cls in range(1, num_classes):
        # 创建当前类别的二值掩膜
        true_cls = (y_true == cls).astype(int)
        pred_cls = (y_pred == cls).astype(int)
        
        # 计算当前类别的IoU
        iou = calculate_iou(true_cls, pred_cls)
        ious.append(iou)
    
    return np.mean(ious)

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        iou = metric.binary.jc(pred, gt)
        return dice, hd95, iou
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 0
    else:
        return 0, 0, 0

from segmentation_mask_overlay import overlay_masks
from PIL import Image
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
# matplotlib.use("Agg")  # 使用非交互式后端，适合批量生成图像

def test_single_volume_ori(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    # print(image.shape,label)
    # image, label = image.cpu().detach().numpy(), label.cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float()#.cuda()
            net.eval()
            with torch.no_grad():

                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs[-1], dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    # if test_save_path is not None:
    #     img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    #     prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    #     lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    #     img_itk.SetSpacing((1, 1, z_spacing))
    #     prd_itk.SetSpacing((1, 1, z_spacing))
    #     lab_itk.SetSpacing((1, 1, z_spacing))
    #     sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
    #     sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
    #     sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list


def test_single_volume_inference(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    # image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    image, label = image.squeeze(0), label.squeeze(0)
    # print(image.shape,label)
    # image, label = image.cpu().detach().numpy(), label.cpu().detach().numpy()
    mask_labels = np.arange(1,classes)
    cmaps = mcolors.CSS4_COLORS
    my_colors=['red','darkorange','yellow','forestgreen','blue','purple','magenta','cyan','deeppink', 'chocolate', 'olive','deepskyblue','darkviolet']
    cmap = {k: cmaps[k] for k in sorted(cmaps.keys()) if k in my_colors[:classes-1]}
    if len(image.shape) == 3:
        # aaa
        prediction = np.zeros_like(label.cpu().numpy())
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice.cpu().numpy(), (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            else:
                slice = slice.cpu().numpy()
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():

                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs[-1], dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
                #### version1
                # lbl = label[ind, :, :]
                # # print(np.unique(lbl))
                # # a
                # masks = []
                # for i in range(1, classes):
                #     masks.append(lbl==i)
                # preds_o = []
                # for i in range(1, classes):
                #     preds_o.append(pred==i)
                # # print(image.shape, np.shape(masks))
                # if ind %10==0:
                #     print('111',np.unique(masks))
                # # a
                # gray_image = image[ind, :, :].reshape(512, 512)
                # # 将灰度图转换为RGB图（复制三次）
                # rgb_image = np.stack([gray_image, gray_image, gray_image], axis=-1)
                # fig_gt = overlay_masks(rgb_image, np.array(masks).reshape(512,512,8), 
                #        labels=mask_labels, colors=cmap, alpha=0.5, return_type='mpl')
                # # fig_gt = overlay_masks(image[ind, :, :].reshape(512,512), np.array(masks).reshape(512,512,8), labels=mask_labels, colors=cmap, alpha=0.1, return_type='mpl')
                # plt.close()
                # # fig_pred = overlay_masks(image[ind, :, :].reshape(512,512), np.array(preds_o).reshape(512,512,8), labels=mask_labels, colors=cmap, alpha=0.1, return_type='mpl')
                # fig_pred = overlay_masks(rgb_image, np.array(preds_o).reshape(512,512,8), 
                #        labels=mask_labels, colors=cmap, alpha=0.5, return_type='mpl')
                # plt.close()
                # # Do with that image whatever you want to do.
                # # fig = plt.figure(figsize=(6,6), dpi=300)
                # # ax = fig.add_subplot(111)
                # # ax.imshow(fig_gt)  # 显示 overlay_img；此时 overlay_img 中灰色背景和彩色区域都存在
                # # ax.axis('off')          # 不显示坐标轴
                # fig_gt.savefig(test_save_path + '/' + case + '_' +str(ind) + '_gt.png', bbox_inches="tight", dpi=300)
                # # fig = plt.figure(figsize=(6,6), dpi=300)
                # # ax = fig.add_subplot(111)
                # # ax.imshow(fig_pred)  # 显示 overlay_img；此时 overlay_img 中灰色背景和彩色区域都存在
                # # ax.axis('off')          # 不显示坐标轴
                # fig_pred.savefig(test_save_path + '/' + case + '_' +str(ind) + '_pred.png', bbox_inches="tight", dpi=300)
                ####
                
                ####version2
                # 直接获取多类别标签图，lbl 中的值已经表示各个类别（0 表示背景，1..classes-1 表示器官）
                # shapeslabe = 224
                # lbl = label[ind, :, :].reshape(shapeslabe, shapeslabe)
                # lbl = lbl.cpu().detach().numpy()
                # if ind % 10 == 0:
                #     print("Unique labels in GT:", np.unique(lbl))

                # # 将原图的灰度图转换为 RGB 图（复制三次）
                # gray_image = image[ind, :, :].reshape(shapeslabe, shapeslabe).cpu().detach().numpy()
                # plt.imsave(test_save_path + '/' + case + '_' +str(ind) + '_gray.png', gray_image, cmap='gray')  # 自动处理浮点数据
                # rgb_image = np.stack([gray_image, gray_image, gray_image], axis=-1)
                # masks = np.stack([lbl == i for i in range(1, classes)], axis=-1)
                # # 直接调用 overlay_masks，用 lbl 作为多类别标签图
                # fig_gt = overlay_masks(rgb_image, masks, labels=mask_labels, colors=cmap, alpha=0.8, return_type='mpl')
                # plt.close()
                # fig_gt.savefig(test_save_path + '/' + case + '_' +str(ind) + '_gt.png', bbox_inches="tight", dpi=300)
                # preds_o = np.stack([pred == i for i in range(1, classes)], axis=-1)
                # fig_pred = overlay_masks(rgb_image, preds_o, labels=mask_labels, colors=cmap, alpha=0.8, return_type='mpl')
                # plt.close()
                # fig_pred.savefig(test_save_path + '/' + case + '_' +str(ind) + '_pred.png', bbox_inches="tight", dpi=300)
                ###
                
                # 如果数值在 [0,1] 内，则先乘以255
                # fig_gt = np.uint8(fig_gt)  # 如果数据已经在 0~255 范围内
                # im = Image.fromarray(fig_gt)
                # im.save(test_save_path + '/' + case + '_' +str(ind) + '_gt.png')
                # fig_pred = np.uint8(fig_pred)  # 如果数据已经在 0~255 范围内
                # im = Image.fromarray(fig_pred)
                # im.save(test_save_path + '/' + case + '_' +str(ind) + '_pred.png')
    else:
        # aaa
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label.cpu().numpy() == i))
    # miou = mean_iou(label.cpu().detach().numpy(), prediction, classes)
    # print('mean_iou',miou)
    
    # if test_save_path is not None:
    #     img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    #     prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    #     lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    #     img_itk.SetSpacing((1, 1, z_spacing))
    #     prd_itk.SetSpacing((1, 1, z_spacing))
    #     lab_itk.SetSpacing((1, 1, z_spacing))
    #     sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
    #     sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
    #     sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    # print(image.shape,label)
    # image, label = image.cpu().detach().numpy(), label.cpu().detach().numpy()
    mask_labels = np.arange(1,classes)
    cmaps = mcolors.CSS4_COLORS
    my_colors=['red','darkorange','yellow','forestgreen','blue','purple','magenta','cyan','deeppink', 'chocolate', 'olive','deepskyblue','darkviolet']
    cmap = {k: cmaps[k] for k in sorted(cmaps.keys()) if k in my_colors[:classes-1]}
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float()#.cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input.cuda())
                out = torch.argmax(torch.softmax(outputs[-1], dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
                #### version1
                # lbl = label[ind, :, :]
                # # print(np.unique(lbl))
                # # a
                # masks = []
                # for i in range(1, classes):
                #     masks.append(lbl==i)
                # preds_o = []
                # for i in range(1, classes):
                #     preds_o.append(pred==i)
                # # print(image.shape, np.shape(masks))
                # if ind %10==0:
                #     print('111',np.unique(masks))
                # # a
                # gray_image = image[ind, :, :].reshape(512, 512)
                # # 将灰度图转换为RGB图（复制三次）
                # rgb_image = np.stack([gray_image, gray_image, gray_image], axis=-1)
                # fig_gt = overlay_masks(rgb_image, np.array(masks).reshape(512,512,8), 
                #        labels=mask_labels, colors=cmap, alpha=0.5, return_type='mpl')
                # # fig_gt = overlay_masks(image[ind, :, :].reshape(512,512), np.array(masks).reshape(512,512,8), labels=mask_labels, colors=cmap, alpha=0.1, return_type='mpl')
                # plt.close()
                # # fig_pred = overlay_masks(image[ind, :, :].reshape(512,512), np.array(preds_o).reshape(512,512,8), labels=mask_labels, colors=cmap, alpha=0.1, return_type='mpl')
                # fig_pred = overlay_masks(rgb_image, np.array(preds_o).reshape(512,512,8), 
                #        labels=mask_labels, colors=cmap, alpha=0.5, return_type='mpl')
                # plt.close()
                # # Do with that image whatever you want to do.
                # # fig = plt.figure(figsize=(6,6), dpi=300)
                # # ax = fig.add_subplot(111)
                # # ax.imshow(fig_gt)  # 显示 overlay_img；此时 overlay_img 中灰色背景和彩色区域都存在
                # # ax.axis('off')          # 不显示坐标轴
                # fig_gt.savefig(test_save_path + '/' + case + '_' +str(ind) + '_gt.png', bbox_inches="tight", dpi=300)
                # # fig = plt.figure(figsize=(6,6), dpi=300)
                # # ax = fig.add_subplot(111)
                # # ax.imshow(fig_pred)  # 显示 overlay_img；此时 overlay_img 中灰色背景和彩色区域都存在
                # # ax.axis('off')          # 不显示坐标轴
                # fig_pred.savefig(test_save_path + '/' + case + '_' +str(ind) + '_pred.png', bbox_inches="tight", dpi=300)
                ####
                
                ####version2
                # 直接获取多类别标签图，lbl 中的值已经表示各个类别（0 表示背景，1..classes-1 表示器官）
                # shapeslabe = 512
                # lbl = label[ind, :, :].reshape(shapeslabe, shapeslabe)
                # # if ind % 10 == 0:
                #     # print("Unique labels in GT:", np.unique(lbl))

                # # 将原图的灰度图转换为 RGB 图（复制三次）
                # gray_image = image[ind, :, :].reshape(shapeslabe, shapeslabe)
                # plt.imsave(test_save_path + '/' + case + '_' +str(ind) + '_gray.png', gray_image, cmap='gray')  # 自动处理浮点数据
                # rgb_image = np.stack([gray_image, gray_image, gray_image], axis=-1)
                # masks = np.stack([lbl == i for i in range(1, classes)], axis=-1)
                # # 直接调用 overlay_masks，用 lbl 作为多类别标签图
                # fig_gt = overlay_masks(rgb_image, masks, labels=mask_labels, colors=cmap, alpha=0.8, return_type='mpl')
                # plt.close()
                # fig_gt.savefig(test_save_path + '/' + case + '_' +str(ind) + '_gt.png', bbox_inches="tight", dpi=300)
                # preds_o = np.stack([pred == i for i in range(1, classes)], axis=-1)
                # fig_pred = overlay_masks(rgb_image, preds_o, labels=mask_labels, colors=cmap, alpha=0.8, return_type='mpl')
                # plt.close()
                # fig_pred.savefig(test_save_path + '/' + case + '_' +str(ind) + '_pred.png', bbox_inches="tight", dpi=300)
                ###
                
                # 如果数值在 [0,1] 内，则先乘以255
                # fig_gt = np.uint8(fig_gt)  # 如果数据已经在 0~255 范围内
                # im = Image.fromarray(fig_gt)
                # im.save(test_save_path + '/' + case + '_' +str(ind) + '_gt.png')
                # fig_pred = np.uint8(fig_pred)  # 如果数据已经在 0~255 范围内
                # im = Image.fromarray(fig_pred)
                # im.save(test_save_path + '/' + case + '_' +str(ind) + '_pred.png')
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            shapeslabe = 224
            lbl = label[ind, :, :].reshape(shapeslabe, shapeslabe)
            # if ind % 10 == 0:
                # print("Unique labels in GT:", np.unique(lbl))

            # 将原图的灰度图转换为 RGB 图（复制三次）
            gray_image = image[ind, :, :].reshape(shapeslabe, shapeslabe)
            plt.imsave(test_save_path + '/' + case + '_' +str(ind) + '_gray.png', gray_image, cmap='gray')  # 自动处理浮点数据
            rgb_image = np.stack([gray_image, gray_image, gray_image], axis=-1)
            masks = np.stack([lbl == i for i in range(1, classes)], axis=-1)
            # 直接调用 overlay_masks，用 lbl 作为多类别标签图
            fig_gt = overlay_masks(rgb_image, masks, labels=mask_labels, colors=cmap, alpha=0.8, return_type='mpl')
            plt.close()
            fig_gt.savefig(test_save_path + '/' + case + '_' +str(ind) + '_gt.png', bbox_inches="tight", dpi=300)
            preds_o = np.stack([pred == i for i in range(1, classes)], axis=-1)
            fig_pred = overlay_masks(rgb_image, preds_o, labels=mask_labels, colors=cmap, alpha=0.8, return_type='mpl')
            plt.close()
            fig_pred.savefig(test_save_path + '/' + case + '_' +str(ind) + '_pred.png', bbox_inches="tight", dpi=300)
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    
    # if test_save_path is not None:
    #     img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    #     prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    #     lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    #     img_itk.SetSpacing((1, 1, z_spacing))
    #     prd_itk.SetSpacing((1, 1, z_spacing))
    #     lab_itk.SetSpacing((1, 1, z_spacing))
    #     sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
    #     sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
    #     sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list