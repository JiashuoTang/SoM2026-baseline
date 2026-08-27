# coding=utf-8
import sys
import argparse
from types import SimpleNamespace
import torch
import torch.nn as nn
from calflops import calculate_flops
from model import WiFo2_model
from main import freeze_wifo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class _Tee:
    def __init__(self, stream, file):
        self.stream = stream
        self.file = file

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)

    
class TaskWrapper(nn.Module):
    def __init__(self, model, task_id):
        super().__init__()
        self.model = model
        self.task_id = task_id

    def forward(self, channel, rgb=None, pos=None):
        if self.task_id == 1:
            return self.model(
                channel,
                mask_ratio=0.0,
                mask_strategy="fre",
                task_id=1,
            )
        if self.task_id == 2:
            return self.model(
                channel,
                rgb=rgb,
                pos=pos,
                mask_ratio=0.5,
                mask_strategy="fre",
                task_id=2,
            )
        if self.task_id == 3:
            return self.model(
                channel,
                rgb=rgb,
                mask_ratio=0.0,
                mask_strategy="fre",
                task_id=3,
            )


def build_args(fastdepth_weights):
    return SimpleNamespace(
        size="tinypro",
        patch_size=4,
        t_patch_size=4,
        no_qkv_bias=0,
        pos_emb="SinCos_3D",
        MoE=False,
        csi_enhance=1,
        rgb_pos_enhance=1,
        fastdepth_weights=fastdepth_weights,
        device_id = '0',
    )


def report_task(name, wrapper, inputs, args):
    freeze_wifo(args, wrapper.model, wrapper.task_id)
    with torch.inference_mode():
        output = wrapper(*inputs)

    flops, macs, params = calculate_flops(
        model=wrapper,
        args=list(inputs),
        print_results=False,
        print_detailed=False,
        output_as_string=False,
    )
    total_params = sum(param.numel() for param in wrapper.model.parameters())
    trainable_params = sum(param.numel() for param in wrapper.model.parameters() if param.requires_grad)
    
    input_shapes = [tuple(tensor.shape) for tensor in inputs]
    output_shape = tuple(output.shape)

    print("=" * 80)
    print(name)
    print(f"Input shapes            : {input_shapes}")
    print(f"Output shape            : {output_shape}")
    print(f"FLOPs                   : {flops:,} ({flops / 1e9:.6f} GFLOPs)\n"
          f"MACs                    : {macs:,} ({macs / 1e9:.6f} GMACs)\n"
          f"Params(total/fine-tuned): {total_params:,} / {trainable_params:,}")


def main():
    log_file = open('flops_log.txt', 'w', buffering=1)
    sys.stdout = _Tee(sys.stdout, log_file)

    parser = argparse.ArgumentParser()
    parser.add_argument("--fastdepth_weights", type=str, default="./weights/FastDepthV2_L1_Best.pth")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device_id", type=str, default="0")
    cli_args = parser.parse_args()

    global device
    device = torch.device(f"cuda:{cli_args.device_id}" if torch.cuda.is_available() else "cpu")

    args = build_args(cli_args.fastdepth_weights)
    model = WiFo2_model(args=args).to(device).eval()
    batch_size = cli_args.batch_size

    task1_channel = torch.randn(batch_size, 2, 24, 8, 128, device=device)

    task2_channel = torch.randn(batch_size, 2, 128, 64, device=device)
    task2_rgb = torch.randn(batch_size, 3, 224, 224, device=device)
    task2_pos = torch.randn(batch_size, 2, device=device)

    task3_channel = torch.randn(batch_size, 2, 32, 64, device=device)
    task3_rgb = task2_rgb

    report_task(
        name="Task 1: LoS/NLoS classification",
        wrapper=TaskWrapper(model, task_id=1).to(device).eval(),
        inputs=(task1_channel,),
        args=args
    )

    report_task(
        name="Task 2: MM-aided channel prediction",
        wrapper=TaskWrapper(model, task_id=2).to(device).eval(),
        inputs=(task2_channel, task2_rgb, task2_pos),
        args=args
    )

    report_task(
        name="Task 3: CSI-aided depth estimation",
        wrapper=TaskWrapper(model, task_id=3).to(device).eval(),
        inputs=(task3_channel, task3_rgb),
        args=args
    )


if __name__ == "__main__":
    main()
