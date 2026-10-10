from .import_packages import *


def get_all_file_paths(root_path, ends_with=("jpg", "jpeg", "png")) -> list:
    file_path_list = []
    # 遍历文件夹
    for dir_path, dir_names, file_names in os.walk(root_path):
        for file_name in file_names:
            if file_name.endswith(tuple(ends_with)):
                full_path = os.path.join(dir_path, file_name)
                file_path_list.append(full_path)
    return file_path_list


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total:,}  ({total / 1e6:.2f} M)")
    print(f"可训练量: {trainable:,}  ({trainable / 1e6:.2f} M)")
    # 顺便按子模块拆开看
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:15s}: {n:>12,}  ({n / 1e6:.2f} M)")
