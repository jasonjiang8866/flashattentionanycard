# Triton tutorial acknowledgements

The grouped tile ordering and persistent grouped scheduling patterns in
`bo_kernel_tuner/kernels/gemm.py` and `moe_grouped_gemm.py` are adapted from
the Triton matrix multiplication and grouped GEMM tutorials. Their
MIT license notice is retained below. The remaining project is covered by
the repository's `LICENSE`.

Sources:
- https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html

Copyright 2018-2020 Philippe Tillet

Copyright 2020-2022 OpenAI

Copyright (c) 2023 - 2025 NVIDIA Corporation & Affiliates. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
