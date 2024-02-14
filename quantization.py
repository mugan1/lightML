from typing import Any, Tuple
import tqdm
import os
import time

import torch
import torch.nn as nn
from torch import Tensor
from torch.ao.quantization import fuse_modules
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
import torchvision


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    r"""
    3x3 convolution with padding
    - in_planes: in_channels
    - out_channels: out_channels
    - bias=False: BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        r"""
         - inplanes: input channel size
         - planes: output channel size
         - groups, base_width: ResNext나 Wide ResNet의 경우 사용
        """
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Basic Block의 구조
        self.conv1 = conv3x3(inplanes, planes, stride)  # conv1에서 downsample
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # short connection
        if self.downsample is not None:
            identity = self.downsample(x)

        # identity mapping시 identity mapping후 ReLU를 적용합니다.
        # 그 이유는, ReLU를 통과하면 양의 값만 남기 때문에 Residual의 의미가 제대로 유지되지 않기 때문입니다.
        out += identity
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4 # 블록 내에서 차원을 증가시키는 3번째 conv layer에서의 확장계수

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # ResNext나 WideResNet의 경우 사용
        width = int(planes * (base_width / 64.)) * groups

        # Bottleneck Block의 구조
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation) # conv2에서 downsample
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        # 1x1 convolution layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 3x3 convolution layer
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # 1x1 convolution layer
        out = self.conv3(out)
        out = self.bn3(out)
        # skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        # default values
        self.inplanes = 64 # input feature map
        self.dilation = 1
        # stride를 dilation으로 대체할지 선택
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        r"""
        - 처음 입력에 적용되는 self.conv1과 self.bn1, self.relu는 모든 ResNet에서 동일
        - 3: 입력으로 RGB 이미지를 사용하기 때문에 convolution layer에 들어오는 input의 channel 수는 3
        """
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        r"""
        - 아래부터 block 형태와 갯수가 ResNet층마다 변화
        - self.layer1 ~ 4: 필터의 개수는 각 block들을 거치면서 증가(64->128->256->512)
        - self.avgpool: 모든 block을 거친 후에는 Adaptive AvgPool2d를 적용하여 (n, 512, 1, 1)의 텐서로
        - self.fc: 이후 fc layer를 연결
        """
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, # 여기서부터 downsampling적용
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        r"""
        convolution layer 생성 함수
        - block: block종류 지정
        - planes: feature map size (input shape)
        - blocks: layers[0]와 같이, 해당 블록이 몇개 생성돼야하는지, 블록의 갯수 (layer 반복해서 쌓는 개수)
        - stride와 dilate은 고정
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        # the number of filters is doubled: self.inplanes와 planes 사이즈를 맞춰주기 위한 projection shortcut
        # the feature map size is halved: stride=2로 downsampling
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 블록 내 시작 layer, downsampling 필요
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion # inplanes 업데이트
        # 동일 블록 반복
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    
class QuantizableBasicBlock(BasicBlock):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_relu = torch.nn.quantized.FloatFunctional()
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add_relu.add_relu(out, identity)

        return out

    def fuse_model(self) -> None:
        fuse_modules(self, [["conv1", "bn1", "relu"], ["conv2", "bn2"]], inplace=True)
        if self.downsample:
            fuse_modules(self.downsample, ["0", "1"], inplace=True)

class QuantizableResNet(ResNet):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(QuantizableResNet, self).__init__(*args, **kwargs)

        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        # Ensure scriptability
        # super(QuantizableResNet,self).forward(x)
        # is not scriptable
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self) -> None:
        r"""Fuse conv/bn/relu modules in resnet models

        Fuse conv+bn+relu/ Conv+relu/conv+Bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """
        fuse_modules(self, ["conv1", "bn1", "relu"], inplace=True)
        for m in self.modules():
            if type(m) is QuantizableBasicBlock:
                m.fuse_model()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target):
    """ Computes the top 1 accuracy """
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_one = correct[:1].view(-1).float().sum(0, keepdim=True)
        return correct_one.mul_(100.0 / batch_size).item()
    
def train(net, data_loader, epoch):

    net = net.to(device).train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters())

    running_loss = AverageMeter('loss')
    acc = AverageMeter('train_acc')

    for images, target in tqdm.tqdm(data_loader):
        images = images.to(device)
        target = target.to(device)

        preds = net(images)
        loss = criterion(preds, target)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        running_loss.update(loss.item(), preds.shape[0])
        acc.update(accuracy(preds, target), preds.shape[0])
        
    print('Epoch : %d ' %(epoch + 1), running_loss, acc)
      

def test(net, data_loader, device):

    net = net.to(device).eval()
    num_correct = 0
    num_samples = 0
    total_time = 0
    torch.save(net.state_dict(), "tmp.pth")
    model_size = os.path.getsize("tmp.pth") / 1e6
    os.remove("tmp.pth")
    for images, target in tqdm.tqdm(data_loader):
        images = images.to(device)
        target = target.to(device)
        cur = time.time()
        preds = net(images)
        total_time += time.time() - cur

        preds = preds.argmax(1)

        num_correct += (preds == target).sum()
        num_samples += preds.size(0)

    print(f"Acc: {num_correct / num_samples:.4f}, Inference Time: {total_time / num_samples}, "
          f"Model Size: {model_size:.2f}MB")

def print_size_of_model(model):
    """ Prints the real size of the model """
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def load_model(model, model_file):

    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model = model.cpu()
    return model
    
def measure_inference_latency(model,
                              device,
                              input_size=(1, 3, 32, 32),
                              num_samples=100,
                              num_warmups=10):

    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave

def run_benchmark(model, model_file, img_loader, device):

    elapsed = 0
    # state_dict = torch.load(model_file)
    # model.load_state_dict(state_dict)
    model = model.to(device).eval()
    num_batches = 5
    # 이미지 배치들 이용하여 스크립트된 모델 실행
    for i, (images, target) in enumerate(img_loader):
        images= images.to(device)
        target= target.to(device)
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
    dataset=torchvision.datasets.CIFAR10("./data", True, transforms.Compose([transforms.Resize(224), transforms.ToTensor()]), download=True),
    batch_size=64
    )
    test_loader = DataLoader(
        dataset=torchvision.datasets.CIFAR10("./data", False, transforms.Compose([transforms.Resize(224), transforms.ToTensor()]), download=True),
        batch_size=1
    )

    fp32_model = ResNet(BasicBlock, [2,2,2,2])
    fp32_model.to(device)
    print("Size of FP32 Model")
    print_size_of_model(fp32_model)
    # for i in range(20):
    #     train(fp32_model, train_loader, i)
    # print("Float Model Test ! ")
    # test(fp32_model, test_loader, device)
        
    model_dir = "./models/"
    float_filename = "resnet18_basicblock_cifar10.pt"
    qat_filename = "resnet18_basicblock_cifar10_quant.pt"
    # scripted_float_file = 'resnet_fp32_scripted.pth'
    # scripted_quantized_file = 'resnet_quant_scripted.pth'
    float_model_filepath = os.path.join(model_dir, float_filename)
    qat_model_filepath = os.path.join(model_dir, qat_filename)

    fp32_model.cpu()
    # torch.save(fp32_model.state_dict(), float_model_filepath)
    # torch.jit.save(torch.jit.script(fp32_model), model_dir + scripted_float_model_file)

    fp32_model = load_model(fp32_model, model_dir + float_filename)

    qat_model = QuantizableResNet(QuantizableBasicBlock, [2,2,2,2])
    qat_model.load_state_dict(fp32_model.state_dict())
    device_of_model = next(qat_model.parameters()).device
    print("QAT Model is on device:", device_of_model)

    qat_model.eval()
    qat_model.qconfig = torch.quantization.get_default_qconfig("x86")
    qat_model.fuse_model()

    qat_model = qat_model.cuda().train()
    qat_model = torch.quantization.prepare_qat(qat_model)

    # for i in range(20):
    #     train(qat_model, train_loader, i)
    #     if i > 3:
    #         qat_model.apply(torch.ao.quantization.disable_observer)
    #     if i > 2:
    #         qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    qat_model.cpu()
    # torch.save(qat_model.state_dict(), qat_model_filepath)
    qat_model = load_model(qat_model, model_dir + qat_filename)
    qat_model = qat_model.eval()
    # torch.jit.save(torch.jit.script(qat_model), model_dir+scripted_quantized_file)

    qat_model = torch.quantization.convert(qat_model)

    print("Float Model GPU Test ! ")
    test(fp32_model, test_loader, device)
    fp32_gpu_inference_latency = measure_inference_latency(model=fp32_model, device=device, input_size=(1,3,32,32), num_samples=100)

    cpu_device = torch.device("cpu")

    print("Float Model CPU Test ! ")
    test(fp32_model, test_loader, cpu_device)
    print("QAT Model CPU Test ! ")
    test(qat_model, test_loader, cpu_device)

    fp32_cpu_inference_latency = measure_inference_latency(model=fp32_model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    int8_cpu_inference_latency = measure_inference_latency(model=qat_model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)

    print("FP32 GPU Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))
    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
    print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))

    # run_benchmark(fp32_model, float_model_filepath, test_loader, device)
    # run_benchmark(fp32_model, float_model_filepath, test_loader, cpu_device)
    # run_benchmark(qat_model, qat_model_filepath, test_loader, cpu_device)
