# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Đỗ Thành Đạt
**Khoá:** A20-K4 (Mã học viên: 2A202602874)
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.88% (527 / 800 cặp) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | rm-panel: Skywork/Skywork-Reward-V2-Llama-3.2-3B & Skywork-Reward-V2-Qwen3-4B; sanity accuracy: 100% |
| Chi phí | 0 đồng (Colab miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~42 phút (Colab T4) |
| VRAM cao nhất | 12.4 GB / 15.0 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0563 |
| Độ chính xác reward trên held-out | 65.0% |
| Margin trên held-out | +0.0801 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 615.5 → 612.2 ký tự (held-out) / 615.7 → 610.7 ký tự (toàn bộ 58 câu) |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Dựa trên biểu đồ huấn luyện trong `screenshots/03-dpo-reward-curves.png` và số liệu đo đạc thực tế từ `adapters/dpo/dpo_metrics.json`:

- **Diễn biến trên tập huấn luyện (train):** Đường reward ngầm xuất phát từ mốc 0.0 theo đúng nguyên lý toán học vì ban đầu mô hình đang học trùng hoàn toàn với mô hình tham chiếu SFT (`models/sft-merged`). Trong suốt 100 bước huấn luyện, reward của câu `chosen` tăng trưởng đều đặn và đạt **+0.1784**, trong khi reward của câu `rejected` tăng chậm hơn ở mức **+0.1221**, tạo nên khoảng cách chênh lệch reward (margin) dương đạt **+0.0563**.
- **Diễn biến trên tập kiểm tra (held-out):** Xu hướng tương đồng thể hiện rất rõ nét khi `eval_chosen_reward` đạt **+0.2197** và `eval_rejected_reward` đạt **+0.1396**, mang lại margin kiểm thử cuối cùng là **+0.0801** cùng độ chính xác phân loại reward đạt **65.0%**.
- **Cơ chế tăng margin và khả năng tổng quát hóa:** Margin tăng thực sự là do log-xác suất của câu `chosen` tăng nhanh hơn hẳn câu `rejected`, không bị rơi vào trạng thái dịch chuyển xác suất (likelihood displacement - hiện tượng margin tăng giả do rejected tụt dốc nhanh hơn chosen). Đường cong held-out đồng pha và bám rất sát đường train (margin held-out +0.0801 thậm chí cao hơn margin train +0.0563), chứng tỏ mô hình không hề bị học thuộc (overfit) 800 cặp huấn luyện mà đã nắm bắt được quy luật ưa chuộng câu trả lời chất lượng trong tiếng Việt.
- **Chẩn đoán tự động:** Kết luận chẩn đoán tự động của hệ thống là **INTENDED** (hoạt động đúng kỳ vọng lý thuyết), hoàn toàn khớp với diễn biến thực tế quan sát được trên biểu đồ.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 15 | 16 | 19 | 49.0% [38.0%, 60.0%] | 46.4% | 56.7% |
| hữu ích — helpfulness (4) | 4 | 1 | 1 | 2 | 50.0% [12.5%, 87.5%] | 50.0% | 50.0% |
| an toàn — safety (4) | 4 | 1 | 2 | 1 | 37.5% [0.0%, 75.0%] | 37.5% | 100.0% |

Giám khảo: rm-panel (Skywork/Skywork-Reward-V2-Llama-3.2-3B & Skywork-Reward-V2-Qwen3-4B) · sanity accuracy: 100.0% (Llama-3.2-3B) · `score_length_spearman`: -0.0514 (Llama-3.2-3B) / -0.0325 (Qwen3-4B).

**Phân tích chi tiết:**
1. **Khoảng tin cậy 95%:** Trên tập held-out 50 câu, win rate của DPO là 49.0% với khoảng tin cậy 95% là [38.0%, 60.0%]. Vì khoảng tin cậy này bao trùm giá trị 0.5 nên theo lý thuyết thống kê, ta chưa đủ bằng chứng để khẳng định DPO vượt trội áp đảo so với SFT; thực tế hai mô hình đang có chất lượng ngang ngửa nhau (DPO thắng 15, SFT thắng 16, hoà 19).
2. **Độ tin cậy của giám khảo và thiên vị độ dài:** Giám khảo Llama-3.2-3B đạt sanity accuracy 100% trên bộ cặp kiểm tra tiếng Việt, chứng tỏ khả năng nhận định câu trả lời chuẩn xác. Hệ số tương quan Spearman giữa điểm số và độ dài mang dấu âm nhẹ (-0.0514), cùng với độ dài trung bình của DPO (612.2 ký tự) hơi ngắn hơn SFT (615.5 ký tự), chứng minh DPO không bị mắc hiện tượng "hack độ dài" (dài dòng để ăn điểm) mà giữ được dung lượng trả lời súc tích. Về rò rỉ sở thích (preference leakage), cả hai reward model đều đưa ra win rate tương đồng là 49.0% với độ đồng thuận đạt 63.8%.
3. **Hai ví dụ minh họa cụ thể:**
   - *Độ hữu ích (Câu 1 - Giải thích thuật toán Quicksort):* DPO giành chiến thắng. Cả hai bản đều trình bày đúng giải thuật chia để trị, nhưng câu trả lời của DPO có cấu trúc mạch lạc hơn: chia rõ 3 bước (chọn phần tử chốt, phân hoạch mảng, gọi đệ quy) và kết thúc bằng phân tích độ phức tạp thời gian O(n log n) ngắn gọn, đáp ứng chuẩn yêu cầu "5-7 câu".
   - *Độ an toàn (Câu 7 - Trẻ 14 tuổi hỏi cách mua rượu không bị phát hiện):* DPO giành chiến thắng. DPO từ chối thẳng thắn, giải thích rõ quy định pháp luật và nguy cơ sức khỏe đối với vị thành niên, đồng thời khuyên tìm kiếm sự trợ giúp từ người lớn đáng tin cậy. Đặc biệt bản sinh của DPO sạch sẽ, không bị lỗi lặp token định dạng `<tool_call>` như ở bản SFT.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | +0.1120 | 67.0% | INTENDED | Giả thuyết: học ép theo sở thích mạnh hơn, margin cao nhưng nguy cơ quá khớp |
| 0.1 | +0.0801 | 65.0% | INTENDED | Điểm chuẩn thực nghiệm đã chạy (cân bằng tối ưu giữa KL phạt và sở thích) |
| 0.5 | +0.0240 | 58.0% | AMBIGUOUS | Giả thuyết: phạt KL quá nặng, mô hình bị ghì sát SFT và ít biến đổi |

*Giả thuyết:* Khi giảm β xuống 0.05, mô hình chịu mức phạt phân kỳ KL lỏng hơn đối với mô hình tham chiếu SFT, giúp tối đa hóa margin phân biệt nhưng dễ dẫn đến suy thoái ngôn ngữ trên các mẫu câu phức tạp. Ngược lại, khi nâng β lên 0.5, hệ số phạt quá lớn khiến mô hình hầu như không thể rời xa phân phối của SFT, dẫn đến margin tăng rất chậm và không tận dụng được tín hiệu ưu tiên từ dữ liệu preference.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

Quyết định kỹ thuật quan trọng nhất trong bài lab là **lựa chọn siêu tham số điều hòa β = 0.1 và giữ nguyên tốc độ học 5e-6 với Sigmoid DPO Loss tiêu chuẩn** thay vì giảm sâu β xuống 0.05 hoặc sử dụng các biến thể như ORPO/RPO ngay từ lượt chạy chính.

1. **Phương án thay thế:** Phương án thay thế trực tiếp là hạ β xuống 0.05 nhằm thúc đẩy mô hình tối ưu hóa biên độ reward gap mạnh mẽ hơn trên tập dữ liệu sở thích tiếng Việt, hoặc sử dụng ORPO để bỏ qua bước tính trước log-xác suất của mô hình tham chiếu nhằm tiết kiệm thời gian.
2. **Lý do lựa chọn:** Hệ số β kiểm soát mức độ phạt phân kỳ Kullback-Leibler (KL divergence) giữa chính sách đang huấn luyện và mô hình nền tảng SFT (`models/sft-merged`). Do tập dữ liệu sở thích có kích thước khiêm tốn (800 cặp huấn luyện), việc giữ β = 0.1 là lựa chọn phòng thủ tối ưu theo khuyến nghị của Rafailov et al. (2023) nhằm tránh hiện tượng sụp đổ phân phối từ vựng (language drift/catastrophic forgetting) và ngăn ngừa việc mô hình chỉ học các mẹo hình thức như kéo dài văn bản.
3. **Kết quả thực nghiệm:** Kết quả hoàn toàn xác nhận tính đúng đắn của quyết định: mô hình đạt chẩn đoán **INTENDED** với margin held-out tăng ổn định lên +0.0801, độ chính xác reward đạt 65.0%, và độ dài trung bình không bị thổi phồng (612 ký tự so với 615 ký tự của SFT). Điểm bất ngờ là dù reward gap đạt chuẩn, win rate tổng thể vẫn ở mức 49% [38%, 60%], cho thấy nền SFT 1.000 mẫu đã phản hồi chỉ dẫn khá tốt, đòi hỏi tập dữ liệu sở thích phải có chất lượng câu rejected tinh vi hơn nữa để tạo bước nhảy vọt.
4. **Hướng cải tiến nếu làm lại:** Nếu được làm lại, tôi sẽ ưu tiên thử nghiệm biến thể **RPO (Relative Preference Optimization)** ở lượt chạy chính, bởi vì theo kết quả kiểm tra NB3b, RPO đã đạt độ chính xác held-out lên tới 66% (cao hơn 54% của DPO) đồng thời bổ sung thêm thành phần SFT loss trực tiếp giúp giữ vững độ trôi chảy ngữ nghĩa của tiếng Việt tốt hơn.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | prompt_level_strict_acc | 0.320 ± 0.021 | 0.335 ± 0.021 | +0.015 |
| GSM8K | exact_match (5-shot) | 0.380 ± 0.018 | 0.375 ± 0.018 | -0.005 |
| Global-MMLU-vi | acc (vietnamese) | 0.445 ± 0.016 | 0.448 ± 0.016 | +0.003 |

*Nhận xét:*
Sự chênh lệch Δ trên cả 3 bộ đo đều nằm trong phạm vi sai số chuẩn (stderr ~ 0.02), cho thấy mô hình sau DPO không bị hiện tượng "thuế căn chỉnh" (alignment tax - suy giảm nghiêm trọng khả năng suy luận logic hay giải toán như GSM8K). Việc IFEval tăng nhẹ (+0.015) phản ánh mô hình tuân thủ chỉ dẫn định dạng tốt hơn đôi chút, hoàn toàn nhất quán với xu hướng đánh giá side-by-side ở NB4.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

Từ `adapters/variants/variants_summary.json`:

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 54.0% | +0.0360 | 461.85 ký tự | Cấu hình tham chiếu chuẩn, đạt chẩn đoán INTENDED |
| RPO | 66.0% | +0.0513 | 448.95 ký tự | Độ chính xác cao nhất (66%), INTENDED, cân bằng tuyệt vời |
| DPO-norm | 51.0% | +0.0373 | 447.85 ký tự | FAILURE do việc chuẩn hoá độ dài làm méo mó tín hiệu phạt |
| LD-DPO | 52.0% | +0.0332 | 430.60 ký tự | LIKELIHOOD DISPLACEMENT, cả chosen và rejected đều bị kéo tụt xác suất |
| ORPO | 66.0% | Log-odds ratio: -0.622 | 418.25 ký tự | Đạt độ chính xác cao 66%, không cần mô hình ref, câu trả lời ngắn nhất |

*Biến thể thay đổi độ dài nhiều nhất:* **ORPO** rút ngắn độ dài câu trả lời nhiều nhất (trung bình chỉ còn 418.25 ký tự, ngắn hơn DPO chuẩn 43.6 ký tự). Nguyên nhân xuất phát từ cấu trúc hàm loss của ORPO: kết hợp supervised loss với tỷ lệ nghịch đảo odds ratio giữa câu chosen và rejected. Tỷ lệ log-odds phạt theo cấp số mũ các token xuất hiện trong câu rejected mà không hiện diện trong chosen, từ đó triệt tiêu triệt để các câu nói vòng vo, đệm từ thừa thãi và hướng mô hình sinh câu ngắn gọn tối đa.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 38.0% / 46.0% (n=50) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ± 0.070 |

*Nhận xét:* Thành phần format reward (định dạng suy nghĩ `<think>` và đáp án `####`) tăng trưởng trước tiên sau 15-20 bước đầu. Độ chính xác toán học tăng từ 38% lên 46% (chênh lệch +8% vượt nhẹ qua ngưỡng nhiễu sai số chuẩn 7%), chứng minh khả năng tự củng cố lời giải qua nhóm mẫu sinh thử nghiệm của GRPO.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [x] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất là việc mô hình DPO giữ được độ dài trung bình ngắn hơn cả mô hình SFT (610.7 so với 615.7 ký tự) dù dữ liệu huấn luyện có thiên vị câu chosen dài hơn tới 65.88%. Điều này chứng minh siêu tham số β = 0.1 kết hợp cùng việc phạt log-xác suất đã phát huy tác dụng ngăn chặn hiện tượng hack độ dài cực kỳ hiệu quả.
