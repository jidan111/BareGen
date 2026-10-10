from ..base_structs import *

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels + hidden_channels, out_channels=hidden_channels, kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_channels + 2 * hidden_channels, out_channels=hidden_channels,
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=in_channels + 3 * hidden_channels, out_channels=hidden_channels,
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=in_channels + 4 * hidden_channels, out_channels=in_channels, kernel_size=3,
                               stride=1, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.act(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.act(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, in_channels=16, hidden_channels=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(in_channels=in_channels, hidden_channels=hidden_channels)
        self.rdb2 = ResidualDenseBlock(in_channels=in_channels, hidden_channels=hidden_channels)
        self.rdb3 = ResidualDenseBlock(in_channels=in_channels, hidden_channels=hidden_channels)
        self.norm = channels_get_norms(in_channels)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = self.norm(out)
        return out * 0.2 + x


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode="inter"):
        super(UpSampleBlock, self).__init__()
        self.up_mode = up_mode
        if up_mode == "inter":
            self.up = Interpolate(in_channels=in_channels, out_channels=in_channels, scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2,
                                         padding=1)
        self.out_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                  padding=1)


    def forward(self, x):
        x = self.up(x, scale_factor=2.0, mode="nearest")
        x = self.out_conv(x)
        return x


def split_image_block(tensor, rows, cols, return_cat=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    channel, high, width = tensor.shape
    h_row, h_div = divmod(high, rows)
    w_col, w_div = divmod(width, cols)
    if return_cat:
        assert h_div == 0 and w_div == 0, f"无法被均分为{rows}行{cols}列"
    out = []
    true_col = 0
    for h in range(0, high, h_row):
        true_col = 0
        for w in range(0, width, w_col):
            out.append(tensor[:, h:h + h_row, w:w + w_col].unsqueeze(0))
            true_col += 1
    if return_cat:
        return torch.cat(out, dim=0).to(device), true_col
    return out, true_col


def combine_image_block(arr, cols):
    row_tensor = []
    for i in range(0, len(arr), cols):
        row_tensor.extend(torch.cat(arr[i:i + cols], dim=3))
    tensor = torch.cat(row_tensor, dim=1)
    return tensor
