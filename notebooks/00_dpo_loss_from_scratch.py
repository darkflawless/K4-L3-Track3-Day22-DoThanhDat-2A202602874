# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB0 — DPO loss tự cài từ đầu (CPU, ~10 phút)
#
# **Không cần GPU.** Trước khi gọi `DPOTrainer`, bạn tự viết loss và kiểm tra nó
# trên số liệu đồ chơi. Phần này lấy từ lab K3 (tự cài DPO) và là nền để đọc
# đường cong reward ở NB3.
#
# Bạn sẽ thấy:
# 1. Tại bước 0 (mô hình đang học (policy) = reference) loss luôn bằng `log 2 ≈ 0.693`.
# 2. Gradient của DPO bị nhân với `sigmoid(-margin)`: cặp đã phân biệt tốt gần như không còn được học.
# 3. **Likelihood displacement**: loss vẫn giảm khi log-prob của *chosen* giảm, miễn rejected giảm nhanh hơn.
# 4. IPO, RPO, SimPO, ORPO khác DPO ở đâu, trên cùng một bộ số.

# %%
import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "lab22" / "config.py").exists())
sys.path.insert(0, str(ROOT))

import math

import torch

from lab22 import dpo_math as M

torch.manual_seed(0)

# %% [markdown]
# ## 1. Log-prob của một câu trả lời
#
# `log π(y|x) = Σ_t log π(y_t | x, y_<t)`, chỉ cộng trên token của câu trả lời
# (mask = 1), không cộng trên câu hỏi.

# %%
vocab, length = 8, 5
logits = torch.randn(1, length, vocab)
labels = torch.randint(0, vocab, (1, length))
mask = torch.tensor([[0, 0, 1, 1, 1]])  # 2 token prompt, 3 token trả lời
total, mean = M.sequence_logps(logits, labels, mask)
print(f"sum log p = {total.item():.3f}   mean log p = {mean.item():.3f}")

# %% [markdown]
# ## 2. Bài tập: tự viết DPO loss
#
# Công thức (Rafailov et al. 2023):
#
# $$\mathcal{L} = -\log\sigma\Big(\beta\big[(\log\pi_\theta(y_w) - \log\pi_{ref}(y_w)) - (\log\pi_\theta(y_l) - \log\pi_{ref}(y_l))\big]\Big)$$
#
# Điền hàm dưới đây. Ô kiểm tra sẽ so với bản tham chiếu trong `lab22/dpo_math.py`.


# %%
def my_dpo_loss(pc, pr, rc, rr, beta=0.1):
    """pc/pr: policy log-prob chosen/rejected; rc/rr: reference. Trả về loss trung bình."""
    chosen_reward = beta * (pc - rc)
    rejected_reward = beta * (pr - rr)
    loss = -torch.nn.functional.logsigmoid(chosen_reward - rejected_reward)
    return loss.mean()


# %%
pc, pr = torch.tensor([-12.0, -30.0]), torch.tensor([-15.0, -28.0])
rc, rr = torch.tensor([-13.0, -29.0]), torch.tensor([-14.0, -29.0])
ref_loss, _, _ = M.dpo_loss(pc, pr, rc, rr, beta=0.1)
mine = my_dpo_loss(pc, pr, rc, rr, beta=0.1)
if mine is None:
    print(f"Chưa cài my_dpo_loss. Đáp số tham chiếu: {ref_loss.item():.4f}")
else:
    assert torch.allclose(torch.as_tensor(mine), ref_loss, atol=1e-6), (mine, ref_loss)
    print(f"✓ Khớp tham chiếu: {ref_loss.item():.4f}")

# %% [markdown]
# ## 3. Bước 0: mô hình đang học (policy) = reference ⇒ loss = log 2
#
# NB3 khởi tạo mô hình đang học (policy) bằng chính mô hình SFT (LoRA mới có trọng số B = 0), nên
# reward ngầm định ban đầu bằng 0 và loss bắt đầu ở 0.693. Nếu log của bạn
# không bắt đầu gần 0.693, reference đang không phải mô hình SFT.

# %%
same = torch.tensor([-20.0, -35.0])
loss0, cr0, rr0 = M.dpo_loss(same, same - 3, same, same - 3)
print(f"loss at init = {loss0.item():.4f}   log 2 = {math.log(2):.4f}   rewards = {cr0.tolist()}, {rr0.tolist()}")

# %% [markdown]
# ## 4. Trọng số gradient = sigmoid(−margin)

# %%
for margin in (-2.0, 0.0, 2.0, 5.0):
    m = torch.tensor(margin, requires_grad=True)
    loss = -torch.nn.functional.logsigmoid(m)
    loss.backward()
    print(f"margin {margin:+.1f}: loss {loss.item():.3f}   |dL/dmargin| {abs(m.grad.item()):.3f}")

# %% [markdown]
# ## 5. Likelihood displacement bằng số
#
# Hai kịch bản đều làm margin tăng 2 nat. Loss giống hệt nhau, nhưng ở kịch
# bản B log-prob của câu *được chọn* lại giảm. DPO không phân biệt được hai
# trường hợp này; chỉ đường cong `rewards/chosen` ở NB3 cho bạn biết.

# %%
ref_c, ref_r = torch.tensor([-20.0]), torch.tensor([-22.0])
scenarios = {
    "A: chosen ↑, rejected ↓": (ref_c + 1, ref_r - 1),
    "B: chosen ↓, rejected ↓↓": (ref_c - 3, ref_r - 5),
}
for name, (pc_, pr_) in scenarios.items():
    loss, cr, rj = M.dpo_loss(pc_, pr_, ref_c, ref_r, beta=1.0)
    print(f"{name:28s} loss {loss.item():.3f}  reward chosen {cr.item():+.1f}  rejected {rj.item():+.1f}")

# %% [markdown]
# **RPO** thêm NLL của câu chosen vào loss: kịch bản B bị phạt vì chosen bị đẩy xuống.

# %%
for name, (pc_, pr_) in scenarios.items():
    nll = -pc_ / 10  # NLL trung bình trên 10 token
    print(f"{name:28s} RPO loss {M.rpo_loss(pc_, pr_, ref_c, ref_r, nll, beta=1.0).item():.3f}")

# %% [markdown]
# ### Trả lời câu hỏi về dịch chuyển xác suất (Likelihood Displacement)
#
# **Câu hỏi (README & Rubric):** Vì sao margin có thể tăng trong khi log-xác suất của câu `chosen` lại giảm?
#
# **Trả lời:**
# 1. **Về mặt công thức:**
#    Margin ngầm định của DPO được định nghĩa:
#    $$\Delta = \left(\log \pi_\theta(y_w) - \log \pi_{ref}(y_w)\right) - \left(\log \pi_\theta(y_l) - \log \pi_{ref}(y_l)\right) = \frac{1}{\beta} \left(\hat{r}_\theta(y_w) - \hat{r}_\theta(y_l)\right)$$
#    Hàm loss của DPO là:
#    $$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma(\beta \Delta)\right]$$
#    Hàm loss chỉ phụ thuộc vào hiệu số (margin) $\Delta$, không ràng buộc độc lập giá trị tuyệt đối của từng thành phần $\log \pi_\theta(y_w)$ hay $\log \pi_\theta(y_l)$.
#
# 2. **Cơ chế dịch chuyển (Likelihood Displacement):**
#    - Khi tối ưu loss, gradient cố gắng đẩy margin $\Delta$ tăng lên.
#    - Nếu log-xác suất của câu `chosen` bị giảm ($\log \pi_\theta(y_w) < \log \pi_{ref}(y_w)$), nhưng log-xác suất của câu `rejected` lại bị giảm **mạnh hơn rất nhiều** ($\log \pi_\theta(y_l)$ giảm sâu hơn $\log \pi_\theta(y_w)$), thì hiệu số $\Delta$ vẫn là một số dương lớn.
#    - Xem kịch bản B ở trên: $\log \pi$ của chosen giảm 3 nat, nhưng rejected giảm tới 5 nat $\rightarrow$ Margin vẫn đạt $+2$ nat, và loss giảm hoàn toàn giống kịch bản A (nơi chosen tăng 1 nat, rejected giảm 1 nat).
#
# 3. **Ý nghĩa thực tế:**
#    Mô hình thường dễ tìm ra hướng gradient triệt tiêu các token xấu (rejected) trên toàn bộ phân phối hơn là nâng xác suất token tốt (chosen). Vì vậy, chỉ nhìn thấy margin tăng là chưa đủ để khẳng định mô hình tiến bộ; cần theo dõi riêng hai đường cong `rewards/chosen` và `rewards/rejected` (ở NB3) hoặc bổ sung thành phần NLL (như RPO) để neo xác suất của câu `chosen`.

# %% [markdown]
# ## 6. Bốn biến thể trên cùng một cặp
#
# | Loss | Cần mô hình tham chiếu (reference)? | Chuẩn hoá độ dài? | Ghi chú |
# |---|---|---|---|
# | DPO (sigmoid) | có | không | mức cơ sở (baseline) |
# | IPO | có | có (TRL chia theo số token) | hồi quy margin về 1/(2β), chống quá khớp khi dữ liệu gần như tất định |
# | RPO | có | không | DPO + NLL(chosen), giảm likelihood displacement |
# | SimPO | không | có | log-prob trung bình + margin γ |
# | ORPO | không | có | NLL(chosen) + λ·log-odds-ratio, gộp SFT và sở thích vào một bước |
#
# NB3b huấn luyện thật các biến thể này (TRL `loss_type` và `trl.experimental.orpo`).

# %%
n_tokens_c, n_tokens_r = 40, 120  # chosen ngắn, rejected dài
pc_, pr_ = torch.tensor([-48.0]), torch.tensor([-130.0])
rc_, rr_ = torch.tensor([-50.0]), torch.tensor([-128.0])
avg_c, avg_r = pc_ / n_tokens_c, pr_ / n_tokens_r
print(f"DPO   {M.dpo_loss(pc_, pr_, rc_, rr_)[0].item():.4f}")
print(f"IPO   {M.ipo_loss(pc_, pr_, rc_, rr_, n_tokens_c, n_tokens_r).item():.4f}")
print(f"SimPO {M.simpo_loss(avg_c, avg_r).item():.4f}")
print(f"ORPO  {M.orpo_loss(avg_c, avg_r, -avg_c).item():.4f}")

# %% [markdown]
# **Câu hỏi cho REFLECTION §3:** tổng log-prob của câu dài luôn âm hơn câu ngắn.
# Vì sao điều đó khiến DPO gốc dễ thiên vị độ dài, và SimPO/ORPO xử lý bằng cách nào?
# Gợi ý: NB2 in ra tỉ lệ cặp có chosen dài hơn rejected trong dữ liệu tiếng Việt.
#
# ### Phân tích: Thiên vị độ dài & Cách xử lý của SimPO / ORPO
#
# **1. Vì sao DPO gốc thiên vị độ dài?**
# - Log-prob của một câu là tổng log xác suất từng token: $\log \pi(y|x) = \sum_{t=1}^{|y|} \log \pi(y_t | x, y_{<t})$.
# - Do mỗi xác suất $\pi(y_t) \le 1 \Rightarrow \log \pi(y_t) \le 0$, câu càng dài thì tổng log-prob càng âm.
# - Trong DPO gốc, không có cơ chế chuẩn hóa độ dài $|y|$. Nếu dữ liệu preference có phần lớn câu `chosen` dài hơn `rejected` (như dữ liệu UltraFeedback tiếng Việt thường có tỉ lệ chosen dài hơn chiếm khoảng 60-70%), gradient phạt/thưởng sẽ bị thiên lệch theo số lượng token, khiến mô hình học "lối tắt" (shortcut) là viết dài ra để nhận reward ngầm cao hơn mà không thực sự cải thiện chất lượng nội dung.
#
# **2. Cách SimPO và ORPO giải quyết:**
# - **SimPO:** Chuẩn hóa trực tiếp log-prob bằng độ dài câu (average log-prob: $\frac{1}{|y|} \log \pi(y|x)$) và sử dụng target margin cố định $\gamma$, loại bỏ hoàn toàn ảnh hưởng của độ dài chuỗi token.
# - **ORPO:** Kết hợp hàm mất mát NLL của câu `chosen` với tỷ lệ odds ratio được chuẩn hóa độ dài, giúp cân bằng gradient giữa câu ngắn và câu dài mà không cần mô hình tham chiếu (reference model).
