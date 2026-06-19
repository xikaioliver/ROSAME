import math

from torch.utils.data import Dataset
import torch
import torchvision.transforms as T


class ColumnPermute:
    """
    Randomly permute the 5 column cells (each 16x16) within the bottom 5 cell-rows
    of an image tensor shaped (T, C, 96, 80).

    Layout assumptions:
        - Height = 96  = 6 * 16 (6 cell-rows)
        - Width  = 80  = 5 * 16 (5 cell-columns)
        - We keep the top cell-row (pixels [0:16]) fixed.
        - For the bottom 5 cell-rows (pixels [16:96]), we permute the 5 columns of cells.
        - The same column permutation is applied to all T, C and all bottom cell-rows.
    """

    def __init__(self,
                 cell_h: int = 16,
                 cell_w: int = 16,
                 n_cell_rows: int = 6,
                 n_cell_cols: int = 5):
        """
        Args:
            cell_h: Height of each cell in pixels.
            cell_w: Width of each cell in pixels.
            n_cell_rows: Number of cell-rows (vertical).
            n_cell_cols: Number of cell-columns (horizontal).
        """
        self.cell_h = cell_h
        self.cell_w = cell_w
        self.n_cell_rows = n_cell_rows
        self.n_cell_cols = n_cell_cols

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (T, C, H, W) with H=96, W=80.

        Returns:
            Tensor of same shape with columns of bottom 5 cell-rows permuted.
        """
        # Pixel row where bottom 5 cell-rows start.
        # Top cell-row is index 0, so bottom start at 1 * cell_h.
        row_start_px = self.cell_h  # = 16 for your case

        # Random permutation of the 5 cell-columns
        perm = torch.randperm(self.n_cell_cols)

        # Work on a cloned tensor to avoid modifying in-place
        out = x.clone()

        # For each destination column i, copy from source column perm[i]
        for i in range(self.n_cell_cols):
            src_col = perm[i].item()
            dst_slice = slice(i * self.cell_w, (i + 1) * self.cell_w)
            src_slice = slice(src_col * self.cell_w, (src_col + 1) * self.cell_w)

            # Apply on all T, C and bottom 5 cell-rows (rows 16:96)
            out[:, :, row_start_px:, dst_slice] = x[:, :, row_start_px:, src_slice]

        return out


class RoomBallPermute:
    """
    For an image tensor of shape (T, C, 64, 96) with 16x16 cells:

      - Interpret as a 4x6 grid of cells (4 rows, 6 cols).
      - View this as two 2x6 blocks of cells:
            Block 0 (top):    cell-rows 0,1
            Block 1 (bottom): cell-rows 2,3
      - In each 2x6 block, randomly permute the cells in the *bottom* cell-row
        of that block (cell-row 1 within the block), across the 6 columns.
      - The top cell-row of each block is left unchanged.
      - Same permutations are applied across all T and C.
    """

    def __init__(
        self,
        cell_h: int = 16,
        cell_w: int = 16,
        grid_rows: int = 4,
        grid_cols: int = 6,
        block_height: int = 2,
        p: float = 1.0,
    ):
        """
        Args:
            cell_h: Height of each cell in pixels (default 16).
            cell_w: Width of each cell in pixels (default 16).
            grid_rows: Number of cell-rows (default 4).
            grid_cols: Number of cell-cols (default 6).
            block_height: Number of cell-rows per block (default 2).
            p: Probability of applying the transform.
        """
        self.cell_h = cell_h
        self.cell_w = cell_w
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.block_height = block_height
        self.p = p

        # Starting cell-row offsets for each 2x6 block
        # For 4 rows and block_height 2 -> offsets [0, 2]
        self.block_row_offsets = list(range(0, grid_rows, block_height))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (T, C, H, W) with H=64, W=96.

        Returns:
            Tensor of same shape with bottom rows of each 2x6 block permuted.
        """
        if torch.rand(1).item() > self.p:
            return x

        if x.ndim != 4:
            raise ValueError(f"Expected (T, C, H, W), got {x.shape}")

        T, C, H, W = x.shape

        expected_H = self.grid_rows * self.cell_h
        expected_W = self.grid_cols * self.cell_w
        if H != expected_H or W != expected_W:
            raise ValueError(
                f"Unexpected spatial size: got (H={H}, W={W}), "
                f"expected (H={expected_H}, W={expected_W})"
            )

        out = x.clone()

        # For each 2x6 block
        for block_row_offset in self.block_row_offsets:
            # Bottom cell-row (within block) is local index 1 -> global = block_row_offset + 1
            bottom_cell_row = block_row_offset + (self.block_height - 1)

            # Pixel row range for this bottom cell-row
            r_start = bottom_cell_row * self.cell_h
            r_end = (bottom_cell_row + 1) * self.cell_h

            # Random permutation of the 6 columns
            perm = torch.randperm(self.grid_cols)

            for dst_col in range(self.grid_cols):
                src_col = perm[dst_col].item()

                # Pixel column ranges
                dst_c_start = dst_col * self.cell_w
                dst_c_end = (dst_col + 1) * self.cell_w
                src_c_start = src_col * self.cell_w
                src_c_end = (src_col + 1) * self.cell_w

                out[:, :, r_start:r_end, dst_c_start:dst_c_end] = \
                    x[:, :, r_start:r_end, src_c_start:src_c_end]

        return out


class ItemPermute:
    """
    For an image tensor of shape (T, C, 96, 96) with 16x16 cells:
      - View as a 6x6 grid of cells.
      - Partition into 4 blocks of 3x3 cells:
            [0:3, 0:3], [0:3, 3:6],
            [3:6, 0:3], [3:6, 3:6]  (in cell coordinates)
      - For each 3x3 block, randomly permute its 9 cells.
    """

    def __init__(self,
                 cell_h: int = 16,
                 cell_w: int = 16,
                 grid_rows: int = 6,
                 grid_cols: int = 6,
                 block_rows: int = 3,
                 block_cols: int = 3):
        """
        Args:
            cell_h: Height of each cell in pixels (default 16).
            cell_w: Width of each cell in pixels (default 16).
            grid_rows: Number of cell-rows (default 6).
            grid_cols: Number of cell-cols (default 6).
            block_rows: Cell-rows per block (default 3).
            block_cols: Cell-cols per block (default 3).
        """
        self.cell_h = cell_h
        self.cell_w = cell_w
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.block_rows = block_rows
        self.block_cols = block_cols

        # Predefine the four 3x3 blocks in cell coordinates (row_offset, col_offset)
        self.block_offsets = [
            (0, 0),                    # top-left
            (0, self.block_cols),      # top-right
            (self.block_rows, 0),      # bottom-left
            (self.block_rows, self.block_cols),  # bottom-right
        ]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (T, C, H, W) with H=96, W=96.

        Returns:
            Tensor of same shape with 3x3 cell blocks permuted.
        """
        out = x.clone()

        # Process each 3x3 block
        num_cells_block = self.block_rows * self.block_cols  # 9
        for block_row_offset, block_col_offset in self.block_offsets:
            # Random permutation of the 9 cells within this block
            perm = torch.randperm(num_cells_block)

            for dst_flat in range(num_cells_block):
                src_flat = perm[dst_flat].item()

                # Destination cell (in cell coordinates)
                dst_cell_r = block_row_offset + dst_flat // self.block_cols
                dst_cell_c = block_col_offset + dst_flat % self.block_cols

                # Source cell (in cell coordinates)
                src_cell_r = block_row_offset + src_flat // self.block_cols
                src_cell_c = block_col_offset + src_flat % self.block_cols

                # Convert cell coordinates to pixel ranges
                dst_r_start = dst_cell_r * self.cell_h
                dst_r_end   = (dst_cell_r + 1) * self.cell_h
                dst_c_start = dst_cell_c * self.cell_w
                dst_c_end   = (dst_cell_c + 1) * self.cell_w

                src_r_start = src_cell_r * self.cell_h
                src_r_end   = (src_cell_r + 1) * self.cell_h
                src_c_start = src_cell_c * self.cell_w
                src_c_end   = (src_cell_c + 1) * self.cell_w

                out[:, :, dst_r_start:dst_r_end, dst_c_start:dst_c_end] = \
                    x[:, :, src_r_start:src_r_end, src_c_start:src_c_end]

        return out


class BlocksworldPilePermute:
    """
    Randomly permute the block piles for a Blocksworld PDDLGym-style image
    tensor of shape (T, C, 64, 64).

    Assumptions (hard-coded, matching the renderer):

        - Image width  W = 64
        - Image height H = 64

        - Renderer "world" dimensions:
              width, height = 3.2, 3.2
              table_height  = 0.15 * height
              robot_height  = 0.10 * height

          In pixel coordinates, we approximate the vertical span of piles as:
              row_start_px ≈ 0.15 * H  (just above the table)
              row_end_px   ≈ 0.90 * H  (just below the robot)

        - The horizontal axis is split into `num_piles` equal-width vertical
          strips; each strip corresponds to one pile. We randomly permute
          these strips only within [row_start_px : row_end_px].

        - The same permutation is used for all T, C.
    """

    def __init__(self, num_piles: int=5):
        """
        Args:
            num_piles: Number of block piles (horizontal columns) in the scene.
        """
        self.num_piles = int(num_piles)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (T, C, H, W) with H=64, W=64.

        Returns:
            Tensor of same shape with the pile columns permuted in the
            vertical band containing the piles.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected (T, C, H, W), got {x.shape}")

        T, C, H, W = x.shape

        if H != 64 or W != 64:
            raise ValueError(
                f"BlocksworldPilePermute expects (H, W) = (64, 64), got ({H}, {W})"
            )

        # Vertical band where piles live: from just above table to just below robot
        row_start_px = math.floor(0.15 * H)
        row_end_px   = math.ceil(0.90 * H)

        if row_start_px >= row_end_px:
            raise ValueError(
                f"Invalid pile band: row_start_px={row_start_px}, row_end_px={row_end_px}"
            )

        # Width (in pixels) of each pile column
        if self.num_piles <= 0:
            raise ValueError(f"num_piles must be positive, got {self.num_piles}")
        pile_w = W // self.num_piles
        buffer = 1

        # Random permutation of pile indices [0 .. num_piles-1]
        perm = torch.randperm(self.num_piles)

        out = x.clone()

        # For each destination pile i, copy pixels from source pile perm[i]
        for dst_idx in range(self.num_piles):
            src_idx = perm[dst_idx].item()

            # Column ranges in pixels for destination and source
            dst_c_start = dst_idx * (pile_w + buffer)
            dst_c_end   = (dst_idx + 1) * (pile_w + buffer) - buffer
            src_c_start = src_idx * (pile_w + buffer)
            src_c_end   = (src_idx + 1) * (pile_w + buffer) - buffer

            out[:, :, row_start_px:row_end_px, dst_c_start:dst_c_end] = \
                x[:, :, row_start_px:row_end_px, src_c_start:src_c_end]

        return out


class TraceResize:
    def __init__(self, size: int = 64, interpolation=T.InterpolationMode.BILINEAR):
        self.resize = T.Resize((size, size), interpolation=interpolation)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        T_len, C, H, W = x.shape
        frames = [self.resize(x[t]) for t in range(T_len)]
        return torch.stack(frames, dim=0)



class TraceDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.img_traces = data[0]
        self.state_traces = data[1]
        self.action_traces = data[2]
        self.transforms = transforms

    def __len__(self):
        return len(self.img_traces)

    def __getitem__(self, idx):
        if self.transforms is not None:
            return self.transforms(self.img_traces[idx]), self.state_traces[idx], self.action_traces[idx], idx
        else:
            return self.img_traces[idx], self.state_traces[idx], self.action_traces[idx], idx

    def input_shape(self):
        return self.img_traces.shape[1:]