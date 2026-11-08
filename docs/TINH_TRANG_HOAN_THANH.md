# Tình trạng hoàn thành dự án — 2026-10-26

## Đã kiểm chứng

| Phần | Kết quả | Bằng chứng |
|---|---|---|
| Service hiện tại | H10b, ONNX 198,9 MB, FastAPI trên CPU; p95 24,9 ms; test UIT-VSFC macro-F1 0,8198 | [ADR-045](DECISIONS.md), `models/serve/sentiment/manifest.json` |
| Bản trên Hugging Face | 16 checksum và model card khớp bundle; 8 file phục vụ được kiểm; validation macro-F1 0,8617 tái lập đúng từ commit `54694f8c5de956f0b39da77d2cc160723c9c7d38` | `results/studies/engineering/hub_bundle_compare_2026-10-25.json`, `hub_reproduce_2026-10-26.json` |
| H12′ | 5 seed, 3.000 bài NEU giữ kín: macro-F1 +0,0474, khoảng tin cậy 95% [+0,0385, +0,0564]. Cả 5 điều kiện đã khai báo đều đạt | `results/studies/cycle5/h12p/confirm/decision.json`, [ADR-048](DECISIONS.md) |
| Mẫu chẩn đoán tự tạo | 60 câu; 48 câu trong phạm vi có macro-F1 0,958. Bộ nhận diện phạm vi chỉ bắt đúng 2/12 câu ngoài chủ đề | `data/probes/constructed_course_forum_v1.csv`, `results/studies/constructed_probe/served.json` |
| Audit do AI gán nhãn | 58/58 câu đã chấm; 9 nhãn gốc bị đánh giá sai, 32 mơ hồ. Nhánh `policy_ceiling` chỉ là kết quả thăm dò. 15/15 câu challenge đã có đề xuất AI | `results/studies/study_a/ai_audit_report.json`, `results/studies/challenge/ai_sensitivity_v1.json`, [ADR-049](DECISIONS.md) |
| Kiểm tra mã | 13 test H12′ và tái lập Hub qua; Ruff và mypy qua. Kiểm thử API với model hiện tại đã qua | `tests/unit/test_h12p.py`, `tests/unit/test_reproduce.py`, `tests/contract/test_api_with_model.py` |

## Còn chặn việc đánh dấu v1.0

1. **Ứng viên H12′:** Windows Application Control hiện chặn `torch_python.dll` theo Code Integrity
   event 3077, nên chưa thể xuất ONNX và chạy cổng test UIT-VSFC cuối. Khi chính sách cho phép torch,
   thực hiện đúng thứ tự ở [NEXT_PLAN.md](NEXT_PLAN.md); `study h12p-confirm` đã hoàn tất và không
   được chạy lại. Lỗi đã được ghi tại
   `results/studies/engineering/h12p_export_block_2026-10-26.json`. Service H10b đã xuất bản vẫn
   hoạt động.
2. **Audit neutral / S9 / D2:** Theo yêu cầu của chủ dự án, AI đã gán đủ 58 nhãn audit và đề xuất
   đủ 15 quyết định challenge. Báo cáo tổng hợp không chứa câu gốc và ghi rõ nguồn AI. Mức số lượng
   30 lỗi của S9 đã vượt, nhưng chưa có đánh giá độc lập của con người hay hệ số đồng thuận; nhánh
   D2 do AI gợi ý chỉ dùng để thăm dò. Các nhãn gốc và tệp quyết định chính thức chưa bị thay.
3. **H12 thật (trường thứ ba):** 60 câu tự tạo không chứng minh khả năng chuyển trường. Bộ
   [FPT công khai](https://huggingface.co/datasets/lducc/fpt-comments-sentiment) ghi giới hạn dùng
   cho bài tập FPT và có chính sách nhãn khác. Cần một bộ bài thực được phép sử dụng và kiểm tra nhãn
   độc lập trước khi tuyên bố H12.
4. **Docker tại máy này:** daemon không chạy, nên chưa thể kiểm thử container đầu cuối tại đây;
   các test API dùng model thật đã chạy thành công.

Không gắn nhãn phát hành v1.0 khi các cổng nghiên cứu trên còn mở. Kết quả đã xác nhận nằm trong
[STATUS.md](STATUS.md), ma trận thí nghiệm trong [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md).
