# Hướng dẫn các việc chủ dự án làm tay

**Cập nhật:** 2026-10-01, sau quyết định 9 (service chạy H10b, ADR-043). Tài liệu này dành cho chủ dự án. Nó liệt kê những việc máy
không làm thay được, giải thích vì sao cần làm, và hướng dẫn từng bước. Mọi tài liệu khác của dự án
viết bằng tiếng Anh; kế hoạch tổng thể nằm ở [NEXT_PLAN.md](NEXT_PLAN.md).

Các lệnh bên dưới chạy trong PowerShell, tại thư mục gốc của repo
(`D:\GitHub\ViFeedback NLP Service`), với Python của `.venv`:

```powershell
cd "D:\GitHub\ViFeedback NLP Service"
$PY = ".venv\Scripts\python.exe"
```

---

## Tóm tắt

| # | Việc | Thời gian | Khi nào | Nó mở khoá việc gì |
|---|---|---|---|---|
| 1 | [Neutral audit](#1-neutral-audit-gán-nhãn-kiểm-tra-lớp-neutral): [chỉ 58 dòng](#cách-nhanh-chỉ-58-dòng-mà-cây-quyết-định-đọc-adr-044-khuyến-nghị), bảng tính đã sẵn | Khoảng 2 giờ, cộng 30–40 phút sau 24 giờ (bản đầy đủ 160 dòng: 6–8 giờ) | Càng sớm càng tốt | S9 (≥ 30 lỗi được mã hoá), nhánh D2 cho lớp neutral (S4) |
| 2 | [Duyệt 15 nhãn của challenge v1 (U2)](#2-duyệt-15-nhãn-của-challenge-v1-u2) | 20 phút | Bất kỳ lúc nào | Báo cáo challenge v1 kèm nhãn đã được người kiểm |
| 3 | ~~Chuẩn bị máy rảnh để đo độ trễ~~ | — | ✅ Xong 2026-09-30 (bạn cho phép không cần máy rảnh) | p95 12,3 ms (mô hình nhỏ) |
| 4 | ~~Quyết định giấy phép cho H10b~~ | — | ✅ Xong 2026-10-01: H10b lên Hub với CC BY-NC-SA 4.0 | Mô hình nhỏ: chỉ khi bạn muốn đưa lên repo riêng |
| 5 | [Duyệt upload lên Hugging Face](#5-duyệt-upload-lên-hugging-face) | 10 phút mỗi lần | Mỗi khi có bản phát hành mới | Bản trên Hub khớp với service |
| 6 | [Quyết định chính sách nhãn cho trường khác](#6-quyết-định-chính-sách-nhãn-cho-văn-bản-của-trường-khác-quyết-định-số-2) | 15 phút đọc | Trước Cycle 5 (c) | Hướng (c) |
| 7 | [Dữ liệu có nhãn mới cho hướng (c)](#7-dữ-liệu-có-nhãn-mới-cho-hướng-c), hoặc đồng ý chạy [H12′](#phương-án-thay-thế-h12-không-cần-dữ-liệu-mới) | 20–25 giờ (hai người); H12′ chỉ cần một câu trả lời | Trước Cycle 5 (c) | Kết luận về văn bản của trường khác (H12′: chỉ về diễn đàn NEU) |
| 8 | [Challenge v2](#8-tuỳ-chọn-challenge-v2) | 4–6 giờ | Tuỳ chọn | Kiểm tra teencode do người thật gõ |
| 10 | ~~[Chọn mô hình cho service](#10-chọn-mô-hình-cho-service-ba-lựa-chọn)~~ | — | ✅ Xong 2026-10-01: bạn chọn H10b | Service chạy H10b (ADR-043) |
| 11 | [Chạy H12′ trên Kaggle](#11-chạy-h12-trên-kaggle) | 15 phút thao tác, 2–2,5 giờ máy chạy | Bây giờ (laptop đang bị chặn PyTorch) | Kết luận H12′ |

Việc 1 quan trọng nhất: đó là con đường duy nhất để biết lớp `neutral` yếu vì **nhãn gốc mơ hồ**
hay vì **mô hình biểu diễn kém**. Hai nguyên nhân đòi hỏi hai cách sửa khác nhau, và không thí
nghiệm tự động nào phân biệt được chúng.

---

## 1. Neutral audit (gán nhãn kiểm tra lớp neutral)

### Việc này là gì, và vì sao cần

Mô hình yếu nhất ở lớp `neutral`: F1 0.576 trên test UIT-VSFC, trong khi hai lớp kia trên 0.9. Các
thí nghiệm tự động đã loại trừ hai giả thuyết:
- **Mất cân bằng lớp / ngưỡng quyết định:** chỉnh ngưỡng, logit adjustment, cRT đều chỉ nhúc nhích
  macro-F1 cỡ nhiễu (Cycle 1).
- **Bộ phân loại (head):** tương tự, không giúp.

Còn lại hai giả thuyết chỉ con người phân biệt được:
- **Nhãn gốc mơ hồ hoặc sai:** người gán nhãn UIT-VSFC năm 2018 gọi câu góp ý là `negative`, và
  nhiều câu thực sự đọc được theo hai cách. Nếu đúng vậy, mô hình tốt hơn cũng không giúp; phải
  sửa nhãn train hoặc học từ nhãn mềm (soft label).
- **Biểu diễn kém:** nhãn đúng nhưng mô hình không hiểu câu. Khi đó phải đổi encoder hoặc tiền
  huấn luyện thêm.

Bạn đọc 160 câu (train và validation, **không bao giờ có test**), tự gán nhãn, rồi đánh giá nhãn
gốc. Lệnh `study audit-report` sẽ áp dụng một **cây quyết định đã đóng băng từ trước** trong
`configs/experiments/cycle2.yaml`:

| Kết quả trên các câu lỗi quanh lớp neutral | Nhánh | Việc tiếp theo (tự động, không cần bạn) |
|---|---|---|
| ≥ 30% nhãn gốc bị đánh `incorrect` | `incorrect_gold` | E04: huấn luyện lại với nhãn train đã sửa |
| ≥ 40% bị đánh `ambiguous` | `policy_ceiling` | Một lần chạy với nhãn mềm, rồi dừng tối ưu neutral trên benchmark này |
| Còn lại | `representation` | Thử encoder khác (ViSoBERT, BamiBERT) hoặc tiền huấn luyện thêm (TAPT) |

Audit cũng tạo ra ≥ 30 lỗi được mã hoá theo loại, tức mục tiêu S9.

### Bảng tính gồm những gì

File `results/studies/study_a/local/audit_sheet.csv` có 160 dòng. Thư mục `local/` bị git bỏ qua
vì chứa câu gốc của UIT-VSFC. **Không bao giờ commit file này.** Nếu file mất, lệnh
`study audit-sheet` dựng lại y hệt từ các file chỉ số đã commit.

Các dòng được chọn có chủ đích (lấy mẫu phân tầng), nên câu lỗi xuất hiện nhiều hơn tỷ lệ thật:

| Tầng (`stratum`) | Train | Validation | Ý nghĩa |
|---|---:|---:|---|
| `random` | 20 | 20 | Ngẫu nhiên. Chỉ tầng này ước lượng được tỷ lệ trên toàn corpus |
| `error:A->B` (6 loại) | 48 | 42 | Nhãn gốc A, mô hình đoán B |
| `correct_uncertain:*` | 15 | 15 | Mô hình đoán đúng nhưng không chắc |

Thứ tự cột:

| Cột | Bạn làm gì |
|---|---|
| `example_index`, `split` | Khoá dòng, không sửa |
| `stratum` | **Ẩn khi gán nhãn mù**: nó để lộ nhãn gốc và dự đoán |
| `text` | Câu cần đọc |
| `gold` | Nhãn gốc. **Ẩn khi gán nhãn mù** |
| `annotator_label` | **Điền** ở lượt mù |
| `neutral_subtype` | **Điền** ở lượt mù (khi cần, xem bên dưới) |
| `gold_assessment` | **Điền** ở bước 2, sau khi hiện cột `gold` |
| `notes` | Ghi chú tự do, **bắt buộc** khi đánh `incorrect` hoặc chọn subtype `other` |
| các cột sau `notes` | Cờ ngôn ngữ và dự đoán của mô hình. **Ẩn** cho đến khi xong cả hai bước |

### Quy ước của corpus

| Chuỗi | Nghĩa | Cách xử lý |
|---|---|---|
| `wzjwz` + số | Tên người hoặc môn học đã ẩn danh | Là một cái tên, không mang cảm xúc |
| `colonsmile`, `colonlove`, `colonbigsmile` | Biểu tượng cảm xúc tích cực viết thành chữ | Gợi ý giọng điệu, yếu nếu đứng một mình |
| `colonsad`, `coloncontemn` | Biểu tượng cảm xúc tiêu cực | Như trên |
| `doubledot`, `dotdotdot` | Dấu `:` và `...` viết thành chữ | Không mang cảm xúc |

Câu viết thường, thiếu dấu, teencode là một phần của dữ liệu, không phải lỗi cần sửa.

### Quy tắc gán nhãn

Nhãn mô tả **thái độ sinh viên thể hiện với đối tượng** của câu: giảng dạy, chương trình, cơ sở vật
chất, giảng viên.

| Nhãn | Câu... | Ví dụ tự đặt (không lấy từ corpus) |
|---|---|---|
| `positive` | khen, hài lòng, tán thành | *thầy giảng dễ hiểu* · *phòng học mát, sạch sẽ* |
| `negative` | chê, không hài lòng, phàn nàn, nêu vấn đề | *slide chữ quá nhỏ, khó nhìn* · *cô hay đến muộn* |
| `neutral` | không có thái độ đánh giá, hoặc cân bằng / không rõ đến mức không gọi được | *môn học có 3 tín chỉ* · *em không có ý kiến* |
| `ambiguous` | *chỉ dành cho người audit*: hai người đọc cẩn thận có thể bất đồng, và không ai đọc sai | — |

**Câu góp ý là trường hợp khó, và corpus đã quyết định rồi.** 91,1% câu chứa *nên* / *cần* / *mong*
có nhãn gốc `negative`: người gán nhãn gốc coi lời đề nghị thay đổi là phê bình ngầm tình trạng hiện
tại. Audit đo lỗi **theo chính sách của corpus**, không theo chính sách mới, nên:
- góp ý hoặc đề nghị thay đổi → `negative`, subtype `request_suggestion`
  (*nên giảng chậm lại* · *mong khoa mở thêm lớp buổi tối*);
- đề nghị không hàm ý thiếu sót, như câu hỏi hoặc xin thông tin → `neutral`, subtype
  `request_suggestion` (*cho em hỏi lịch thi khi nào*);
- nếu **bạn** thấy một câu góp ý là trung tính, hãy đánh `ambiguous` và ghi vào `notes`. Đừng đánh
  `neutral`: làm vậy biến bất đồng về chính sách thành bất đồng với nhãn gốc.

### Loại neutral (`neutral_subtype`)

Điền khi `annotator_label` là `neutral` hoặc `ambiguous`, **hoặc** khi `gold` là `neutral` (điền ở
bước 2 nếu lúc đó mới thấy). Các trường hợp khác điền `n/a`. Chọn loại **đầu tiên** khớp:

| Subtype | Định nghĩa | Ví dụ tự đặt |
|---|---|---|
| `objective_fact` | Câu nêu sự kiện, không đánh giá | *lớp học vào buổi sáng thứ hai* |
| `no_opinion` | Sinh viên nói không có gì góp ý, hoặc trả lời cho có | *không có ý kiến gì* · *bình thường* (đứng một mình) |
| `request_suggestion` | Đề nghị, câu hỏi, góp ý. Ghi nhận dù nhãn là gì | *cho em hỏi lịch thi khi nào* |
| `mixed` | Có cả khen lẫn chê, không bên nào trội | *thầy nhiệt tình nhưng giảng hơi nhanh* |
| `insufficient_context` | Có từ đánh giá, nhưng không biết nó nói về cái gì hoặc theo hướng nào nếu thiếu ngữ cảnh | *cũng được* · *như vậy là ổn rồi ạ* |
| `other` | Không thuộc loại nào. **Mô tả trong `notes`** | — |

Hãy để ý `mixed`: UIT-VSFC gán một cảm xúc cho cả câu, nên câu vừa khen thầy vừa chê phòng học không
có nhãn đúng duy nhất. Nếu `mixed` chiếm nhiều trong các lỗi neutral, đó là bằng chứng cần phân tích
cảm xúc theo khía cạnh (aspect-level), không phải cần mô hình tốt hơn.

### Đánh giá nhãn gốc (`gold_assessment`)

Chọn giá trị **thận trọng nhất** phù hợp:

| Giá trị | Khi nào |
|---|---|
| `agree` | Nhãn của bạn trùng nhãn gốc, **hoặc** khác nhưng nhãn gốc vẫn là một cách đọc hợp lý |
| `ambiguous` | Câu thực sự đọc được nhiều cách, và nhãn gốc là một trong số đó |
| `incorrect` | Theo mọi cách đọc hợp lý, và theo hướng dẫn này, nhãn gốc sai. **Bắt buộc ghi lý do trong `notes`** |

`incorrect` là khẳng định mạnh; mặc định là **không** dùng nó. Mô hình tự tin cãi nhãn gốc chỉ là lý
do để xem kỹ, không phải bằng chứng nhãn sai. Phân biệt `ambiguous` với `incorrect` chính là điều
audit này đo: `ambiguous` nghĩa là định nghĩa nhiệm vụ chưa đủ rõ, `incorrect` nghĩa là quá trình
gán nhãn đã nhầm. Hai kết quả dẫn tới hai cách sửa khác nhau.

### Các bước làm

**Chuẩn bị (5 phút).**

1. Kiểm tra bảng tính đã có chưa; nếu chưa thì dựng lại (không cần GPU):
   ```powershell
   & $PY -m vifeedback.cli study audit-sheet
   ```
   Lệnh này **ghi đè** `audit_sheet.csv`. Vì vậy bạn làm việc trên một bản sao:
   ```powershell
   Copy-Item results\studies\study_a\local\audit_sheet.csv results\studies\study_a\local\audit_pass1.csv
   ```
2. Mở `audit_pass1.csv` bằng Excel (file có BOM UTF-8 nên tiếng Việt hiển thị đúng).
3. **Ẩn** các cột `stratum`, `gold`, `gold_assessment` và mọi cột bên phải `notes`.

**Bước 1: lượt mù (khoảng 4–5 giờ, nên chia 3–4 buổi).**
- Với từng dòng, chỉ đọc cột `text`, rồi điền `annotator_label`, `neutral_subtype` (hoặc `n/a`) và
  `notes` nếu cần.
- Đánh giá câu như nó được viết. Đừng đoán môn học, giảng viên hay phần còn lại của phiếu khảo sát.
  Nếu không đánh giá được khi thiếu ngữ cảnh, thì đó chính là câu trả lời (`insufficient_context`).
- Làm xong **cả 160 dòng** rồi mới sang bước 2.

**Bước 2: so với nhãn gốc (khoảng 1–2 giờ).**
- Hiện cột `gold` và `gold_assessment`, rồi điền `gold_assessment` cho từng dòng.
- **Không sửa** `annotator_label` đã điền ở bước 1. Nếu nhận ra mình đọc nhầm, ghi vào `notes` và giữ
  nhãn cũ.
- Nếu `gold` là `neutral` mà `neutral_subtype` đang trống, điền subtype.

**Bước 3: lưu.** Trong Excel chọn *File → Save As → CSV UTF-8 (Comma delimited)*, giữ tên
`audit_pass1.csv`. Các cột mô hình chỉ được hiện ra sau bước 3, để phân tích sau.

**Bước 4: lượt thứ hai để đo độ nhất quán (1 giờ, sau ít nhất 24 giờ).**
- Nếu có người thứ hai, nhờ họ làm bước 1 trên một bản sao mới. Nếu không, bạn tự làm lại **sau ít
  nhất 24 giờ**, trên bản sao mới, không mở lại `audit_pass1.csv`:
  ```powershell
  & $PY -m vifeedback.cli study audit-sheet
  Copy-Item results\studies\study_a\local\audit_sheet.csv results\studies\study_a\local\audit_pass2.csv
  ```
- Chỉ cần điền `annotator_label` cho **ít nhất 50 dòng**, trong đó có **cả 40 dòng tầng `random`**.
  Các dòng khác để trống.
- Hai người khác nhau cho *inter-annotator agreement*; một người làm hai lần chỉ cho *intra-annotator*
  (mức nhất quán của một người). Báo cáo sẽ ghi rõ loại nào.

**Bước 5: chạy phân tích (1 phút).**
```powershell
& $PY -m vifeedback.cli study audit-report `
    --sheet results\studies\study_a\local\audit_pass1.csv `
    --second results\studies\study_a\local\audit_pass2.csv `
    --kind intra      # hoặc inter nếu có người thứ hai
```
Lệnh từ chối nếu `audit_pass1.csv` còn ô trống hoặc giá trị lạ (ví dụ `Neutral ` có dấu cách vẫn được
chấp nhận, nhưng `trung tính` thì không). Kết quả, chỉ có số đếm và không có câu nào, được ghi vào
`results/studies/study_a/audit_report.json`: nhánh của cây quyết định, tỷ lệ theo tầng, khoảng tin cậy
Wilson trên tầng `random`, và hệ số κ (Cohen's kappa).

**Sau đó:** báo cho tôi. Tôi commit `audit_report.json`, ghi ADR, và chạy nhánh D2 mà cây quyết định
chỉ ra. Bảng tính không bao giờ được commit.

### Cách nhanh: chỉ 58 dòng mà cây quyết định đọc (ADR-044, khuyến nghị)

**Có tự động hoá hoàn toàn được không?** Không, nếu muốn đảm bảo chất lượng. Audit hỏi đúng câu mà
người đọc cẩn thận còn bất đồng. LLM đồng ý với nhãn gốc không chứng minh nhãn gốc đúng, và không có
nhãn của người thì không kiểm được LLM (Pangakis và cộng sự, 2023).

**Nhưng có cách giảm hai phần ba công sức mà không mất chất lượng.** Cây quyết định chỉ đọc 58 dòng
thuộc bốn tầng lỗi quanh lớp neutral, không đọc cả 160 dòng. Bạn chỉ cần gán 58 dòng đó; nhánh D2 được
quyết hoàn toàn bằng nhãn của người, không có LLM. 58 lỗi được mã hoá cũng đủ cho S9. Phương án A1′
cũ (LLM gán phần còn lại) bị bỏ, vì nó để LLM quyết một phần nhánh.

**Các bước:**

1. **Bảng tính đã có sẵn** (tôi đã tạo): `results\studies\study_a\local\audit_scope_pass1.csv`, 58
   dòng, thứ tự đã xáo nên không đoán được tầng. Mở bằng Excel.
2. **Lượt mù (khoảng 1,5 giờ).** Ẩn mọi cột **bên phải `notes`** (`gold`, `gold_assessment`,
   `stratum` và các cột mô hình). Điền `annotator_label`, `neutral_subtype` (hoặc `n/a`), `notes` nếu
   cần, theo đúng quy tắc ở trên.
3. **So với nhãn gốc (khoảng 30 phút).** Hiện cột `gold` và `gold_assessment`, điền
   `gold_assessment`. Không sửa nhãn ở bước 2.
4. **Lưu** dạng *CSV UTF-8*, giữ tên file.
5. **Lượt hai sau ít nhất 24 giờ (30–40 phút)**, để đo độ nhất quán. Tạo bảng mới với thứ tự khác,
   rồi chỉ điền `annotator_label` (và subtype khi cần):
   ```powershell
   & $PY -m vifeedback.cli study audit-scope-sheet --out results\studies\study_a\local\audit_scope_pass2.csv --order-seed 46
   ```
6. **Báo tôi.** Tôi chạy:
   ```powershell
   & $PY -m vifeedback.cli study audit-report --scope-only `
       --sheet results\studies\study_a\local\audit_scope_pass1.csv `
       --second results\studies\study_a\local\audit_scope_pass2.csv --kind intra
   ```
   rồi commit kết quả (chỉ số đếm, không có câu nào), ghi ADR và chạy nhánh D2.

Lệnh tạo bảng **không bao giờ ghi đè** file đã có, nên công sức của bạn không bị mất. 40 dòng ngẫu
nhiên (để ước lượng tỷ lệ nhãn sai trên toàn corpus) có thể làm sau; cây quyết định không cần chúng.

---

## 2. Duyệt 15 nhãn của challenge v1 (U2)

**Là gì.** Challenge v1 là 305 câu do dự án tự viết để thử mô hình (không phải dữ liệu của trường
nào). Có 15 câu mà cả gpt-4o-mini lẫn Qwen3 đều cãi nhãn đã đóng băng. Có thể các LLM sai, cũng có thể
nhãn sai. Người viết nhãn cần xem lại.

**File:** [`results/studies/challenge/label_review_v1.csv`](../results/studies/challenge/label_review_v1.csv).
Các cột `gpt4o_mini`, `qwen3_1.7b`, `ce`, `augmented` là dự đoán của từng mô hình; `sentiment` là nhãn
đã đóng băng.

**Cách điền.** Với mỗi dòng, điền cột `owner_decision` bằng một trong các giá trị:

| Giá trị | Nghĩa |
|---|---|
| `keep` | Nhãn đúng, giữ nguyên |
| `negative` / `neutral` / `positive` | Nhãn sai; đây là nhãn đúng |
| `ambiguous` | Câu đọc được nhiều cách |

Cột `owner_note` ghi lý do ngắn nếu không phải `keep`. Ví dụ `c135` *"thay chua bao gio vao lop tre"*
(thầy chưa bao giờ vào lớp trễ) có nhãn `positive`, và cả hai LLM đoán `negative`: có lẽ chúng bị từ
"trễ" đánh lừa, nhưng nhãn vẫn đúng → `keep`.

**Điều gì xảy ra sau đó.** Challenge v1 vẫn đóng băng: không sửa file dữ liệu, vì mọi kết quả cũ phải
tái lập được. Tôi sẽ thêm một bảng "độ nhạy" báo cáo challenge v1 trên nhãn đã duyệt bên cạnh nhãn
gốc. Lưu file (CSV UTF-8), rồi báo tôi commit, vì file này không chứa dữ liệu của ai.

---

## 3. Chuẩn bị máy rảnh để đo độ trễ

**Vì sao.** Mục tiêu độ trễ là p95 ≤ 30 ms cho một câu, trên CPU của laptop (Ryzen 5 6600H). Con số
chính thức 26,9 ms được đo khi service còn dùng điểm Mahalanobis. Bộ phát hiện phạm vi mới (ADR-034)
nhẹ hơn (+0,5–0,8 ms so với +1,3–3,1 ms), nhưng ba lần đo sau ADR-034 chạy khi máy đang bận
(VS Code, Task Manager...), nên con số tuyệt đối không dùng được. Cần ba phiên đo trên máy rảnh.

**Các bước.**
1. Cắm sạc. Để Windows ở chế độ nguồn mặc định (*Settings → System → Power → Power mode: Balanced*),
   giống các lần đo trước, và ghi lại chế độ đã dùng.
2. Đóng mọi ứng dụng: VS Code, trình duyệt, Task Manager, Kaggle, OneDrive sync nếu được. Không có
   job huấn luyện nào đang chạy.
3. Mở một cửa sổ PowerShell riêng, đợi 5 phút cho máy nguội, rồi chạy phiên 1 (mất khoảng 8–10 phút):
   ```powershell
   cd "D:\GitHub\ViFeedback NLP Service"
   .venv\Scripts\python.exe -m vifeedback.cli study latency --no-torch `
       --out results\studies\latency\sessions\cycle5_idle_session1.json
   ```
4. Nghỉ 5 phút, chạy phiên 2 (`...session2.json`), nghỉ 5 phút, chạy phiên 3 (`...session3.json`).
5. Mở lại VS Code và báo tôi. Tôi đọc ba file, lấy trung vị, ghi vào ADR và tài liệu.

Không dùng máy trong lúc đo. Chỉ một tab trình duyệt đang mở cũng làm lệch p95.

---

## 4. Quyết định giấy phép khi mô hình học từ ViLexNorm

**Bối cảnh.**
- ViLexNorm dùng giấy phép **CC BY-NC-SA 4.0** (phi thương mại, *chia sẻ tương tự*). Trọng số hiện
  tại mang **CC BY-NC 4.0**.
- Mô hình H10 (b) học từ các cặp câu ViLexNorm, nhưng **không đạt** (ADR-038), nên không phát hành.
- Mô hình nhỏ H11 (a) cũng đọc câu chữ ViLexNorm khi học: câu gốc nằm trong tập chuyển giao
  (transfer set), và mô hình thầy gán nhãn mềm cho chúng.
- Mô hình **H10b**, mà service đang chạy, học trực tiếp từ các cặp câu ViLexNorm.
- Vì vậy câu hỏi dưới đây áp dụng cho cả H10b và H11.

Việc trọng số học từ dữ liệu SA có phải là "tác phẩm phái sinh" hay không chưa được pháp lý làm rõ.
Cách an toàn là: **khi H10b hoặc H11 được đưa lên Hub, trọng số mang CC BY-NC-SA 4.0**. Thay đổi này không
ảnh hưởng mục đích nghiên cứu phi thương mại của bạn.

**Bạn cần trả lời một câu** trước khi upload: đồng ý đổi giấy phép trọng số sang
CC BY-NC-SA 4.0 (khuyến nghị), hay giữ CC BY-NC 4.0 và **không** đưa H10b/H11 lên Hub (service vẫn dùng
được ở máy bạn).

---

## 5. Duyệt upload lên Hugging Face

Mỗi lần upload là một lần phát hành công khai, nên cần bạn duyệt riêng; lần duyệt trước không áp dụng
cho lần sau.

**Quy trình mỗi lần có bản mới:**
1. Tôi tạo bản chạy thử (dry run) vào `models/publish/vifeedback-sentiment-phobert/` và gửi bạn các
   điểm thay đổi.
2. Bạn đọc `README.md` trong thư mục đó (model card): số liệu, giới hạn, giấy phép.
3. Bạn trả lời "duyệt upload". Tôi chạy `serve publish --upload`, sau đó tải bản trên Hub về, kiểm
   từng file theo `SHA256SUMS`, và chạy lại điểm validation (`serve reproduce`).

Token Hugging Face nằm trong `.env` (bị git bỏ qua). Không bao giờ dán token vào chat.

---

## 6. Quyết định chính sách nhãn cho văn bản của trường khác (quyết định số 2)

**Vấn đề.** UIT-VSFC gọi câu góp ý là `negative`. NEU-ESC (diễn đàn của một trường khác) gọi phần
lớn câu hỏi và góp ý là `neutral`. Cycle 4 cho thấy mô hình học một đầu ra cho cả hai chính sách thì
kém đi trên UIT-VSFC. Hai đầu ra riêng (two heads) thì không bị vậy: đầu NEU-ESC đạt 0,76 macro-F1 trên
validation.

**Hai lựa chọn:**

| Lựa chọn | Nghĩa là | Hệ quả |
|---|---|---|
| **A. Giữ chính sách UIT-VSFC** (mặc định hiện tại) | Service luôn trả nhãn theo cách hiểu của UIT-VSFC, kể cả với văn bản trường khác | Góp ý ở mọi trường đều là `negative`; nhất quán với các kết quả cũ |
| **B. Theo chính sách của nguồn** | Service có thêm đầu ra cho văn bản diễn đàn, nhãn theo cách hiểu của NEU-ESC | Cùng một câu góp ý có thể là `negative` hay `neutral` tuỳ đầu ra được gọi. API cần một tham số chọn đầu ra |

Không có lựa chọn "đúng" về kỹ thuật; đây là câu hỏi người dùng service muốn "neutral" nghĩa là gì.
Chỉ cần trước khi bắt đầu Cycle 5 (c).

---

## 7. Dữ liệu có nhãn mới cho hướng (c)

**Vì sao cần.** Tập test NEU-ESC đã được dùng cho H8 và B4′ (có ghi log). Dùng lại nó cho một giả
thuyết mới làm kết luận yếu đi: kết quả khi đó phản ánh việc đã chọn theo tập test. Hiện không có bộ
dữ liệu công khai nào khác về phản hồi sinh viên tiếng Việt có nhãn (đã tìm, 2026-09-29). Vì vậy mọi
kết luận mới về "văn bản của trường khác" cần một mẫu có nhãn mới.

**Cần gì.**
- **Nguồn:** khoảng 600–800 bài/bình luận của sinh viên ở một trường **khác** UIT và NEU (diễn đàn,
  nhóm lớp, khảo sát môn học), thu thập hợp pháp. Xoá tên người và thông tin cá nhân trước khi gán
  nhãn. Văn bản không bao giờ vào git.
- **Người gán nhãn:** hai người, gán độc lập, theo [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) và chính
  sách đã chọn ở việc 6. Khoảng 10–12 giờ mỗi người cho 700 câu, gồm cả chủ đề (in-scope / off-topic).
- **Hoà giải:** câu hai người bất đồng được thảo luận. Nếu vẫn bất đồng thì nhãn là `ambiguous` và câu
  bị loại khỏi đánh giá chính.
- **Cỡ mẫu:** 600 câu in-scope đủ để phát hiện chênh lệch macro-F1 cỡ +0,05 bằng bootstrap ghép cặp.
  Hiệu ứng của Cycle 4 là +0,11.

Khi bạn có nguồn dữ liệu, tôi sẽ viết quy trình chi tiết (mẫu bảng tính, lệnh kiểm tra độ đồng thuận)
và khai báo quy tắc trong `cycle5.yaml` **trước khi** mở dữ liệu.

### Phương án thay thế H12 (không cần dữ liệu mới)

**H12′** (đề xuất, chờ bạn đồng ý) dùng phần dữ liệu đã có:
- Tách một lần, cố định, **3.000 bài in-scope** từ tập train NEU-ESC (21.113 bài). Không mô hình nào
  được so sánh từng được chấm trên phần này.
- Huấn luyện mô hình hai đầu ra (đầu UIT-VSFC được phục vụ, nên **không cần quyết định 2**) trên
  phần train còn lại, thêm phần huấn luyện nhất quán của H10b, rồi so với 5 seed của H10b trên 3.000
  bài đó.
- Quy tắc khai báo trước: điểm trên 3.000 bài phải cao hơn, và không được kém trên UIT-VSFC, lớp
  neutral, câu mất dấu (điểm H8 từng trượt) và câu gõ tắt.
- Khoảng 3 giờ GPU laptop, mỗi lần một job.

**Giới hạn cần hiểu rõ.** 3.000 bài này cùng trường (NEU), cùng diễn đàn, cùng người gán nhãn với
phần huấn luyện. H12′ trả lời "service có tốt hơn trên văn bản kiểu diễn đàn NEU không", **không**
trả lời "có tốt hơn ở trường thứ ba không". Câu hỏi thứ hai vẫn cần dữ liệu mới như mô tả ở trên.

**Bạn chỉ cần trả lời: "đồng ý H12′"** (quyết định 10). Tôi khai báo trong `cycle5.yaml` v6 trước khi
tách dữ liệu, rồi chạy.

---

## 8. (Tuỳ chọn) Challenge v2

Challenge v2 là một bộ câu do **người thật gõ** (teencode, không dấu, viết tắt) với nhãn do người gán,
để kiểm tra điều NEU-ESC không kiểm được: teencode tự nhiên trong đúng văn phong khảo sát môn học.
Nó không chặn việc nào; NEU-ESC đã thay nó làm tập xác nhận (ADR-030). Nếu sau này có người giúp, quy
trình nằm trong [EVALUATION_DATA.md](EVALUATION_DATA.md).

---

## 10. Chọn mô hình cho service: ba lựa chọn

**✅ Xong 2026-10-01.** Bạn chọn **H10b** (ADR-043). Service đang chạy nó; 40 bài kiểm thử API đạt.

| Thư mục | Mô hình |
|---|---|
| `models/serve/sentiment` | **H10b, bản đã cắt vocabulary** (đang phục vụ từ ADR-045), 198,9 MB, T = 1,349 |
| `models/serve/.fp32-sentiment` | Cũng mô hình H10b, đầy đủ vocabulary, FP32, 540 MB |
| `models/serve/.student-sentiment` | Mô hình nhỏ 6 lớp (H11), 185 MB |
| `models/serve/.previous-sentiment` | Mô hình 12 lớp cũ (p9) |

Kết quả so sánh lúc chọn:

| | Mô hình 12 lớp cũ (p9) | Mô hình nhỏ | **H10b (đang phục vụ)** |
|---|---|---|---|
| Kích thước / độ trễ p95 | 540 MB / 22,1 ms | **185 MB / 12,3 ms** | 540 MB / cùng kiến trúc với p9 |
| Macro-F1 trên test (5 seed) | **0,830** | 0,817 (kém rõ rệt) | 0,824 (chênh −0,006, không có ý nghĩa thống kê) |
| Neutral F1 trên test (seed 42) | **0,592** | 0,545 | 0,550 |
| Nhãn đổi khi gõ tắt (ViLexNorm) | khoảng 18% | khoảng 11% | **khoảng 12%** (đã xác nhận theo quy tắc) |
| Challenge set, qua đúng pipeline của service | 0,851, neutral 0,771 | 0,860, neutral 0,779 | **0,916, neutral 0,885** |

**Nếu muốn đổi lại** (PowerShell, tại thư mục repo; không xoá gì), rồi báo tôi để cập nhật tài liệu:
```powershell
# Phục vụ lại H10b bản đầy đủ 540 MB (nếu bạn coi S4 quan trọng hơn S7, xem dưới):
Move-Item models\serve\sentiment models\serve\.trimmed-sentiment
Move-Item models\serve\.fp32-sentiment models\serve\sentiment
# Phục vụ mô hình nhỏ:
Move-Item models\serve\sentiment models\serve\.trimmed-sentiment
Move-Item models\serve\.student-sentiment models\serve\sentiment
```

**Công bố lên Hugging Face.**
- ✅ Bạn đã upload H10b lên repo chính ngày 2026-10-01 (commit trên Hub `037edfb`). Tôi đã tải về,
  kiểm từng file theo `SHA256SUMS` và chạy lại validation: 0,8617, khớp manifest.
- **Bản 198,9 MB đang phục vụ chưa lên Hub** (Hub vẫn là bản 540 MB của cùng mô hình). Bản chạy
  thử đã dựng ở `models/publish/vifeedback-sentiment-phobert/`, card nêu rõ phần cắt vocabulary.
  Nếu muốn Hub khớp service, bạn tự chạy lệnh upload như lần trước (cần bạn duyệt).
- **Đánh đổi cần biết.** Trên test, bản đã cắt khác bản đầy đủ đúng 2 trên 3.166 câu; neutral F1
  0,548 thay vì 0,550, tức hụt mức tối thiểu 0,55 của S4 đúng 2 câu (trong độ nhiễu của một câu).
  Nếu bạn coi mức đó quan trọng hơn dung lượng, dùng lệnh đổi lại ở trên.
- Mô hình nhỏ (tuỳ chọn): bản chạy thử ở `models/publish/vifeedback-sentiment-phobert-6l/`, cho repo
  riêng, giấy phép CC BY-NC-SA 4.0.

---

## 9. Khi Windows chặn thư viện (Application Control)

**Chuyện gì đã xảy ra.**
- Laptop dùng Windows Application Control (Smart App Control), cơ chế chặn các file DLL chưa đủ
  "uy tín".
- Từ lâu, DLL của gói `onnx` đã bị chặn, nên máy này không dựng được đồ thị INT8 (ADR-036).
- Sáng 2026-09-30, sau khi Windows cập nhật (bản 26200 → 26300), DLL của `pyarrow` cũng bị chặn
  trong vài phút. Kéo theo đó, scikit-learn và transformers không nạp được, và dữ liệu UIT-VSFC
  (dạng parquet) không đọc được.
- Vài phút sau, Windows tự cho phép lại: nhiều khả năng nó đang tra cứu uy tín của file mới. Dự án
  **không bao giờ** vượt qua cơ chế này.

**Nếu gặp lại** (lỗi `DLL load failed ... An Application Control policy has blocked this file`):
1. Đợi vài phút rồi thử lại lệnh. Lần 2026-09-30 tự hết.
2. Nếu vẫn bị chặn, mở *Windows Security → App & browser control → Smart App Control* để xem trạng
   thái.
   - Tắt Smart App Control là quyết định về bảo mật máy của bạn.
   - Lưu ý: sau khi tắt thì không bật lại được nếu không cài lại Windows.
3. Nếu không muốn đổi cài đặt, việc huấn luyện có thể chuyển sang Kaggle. Báo tôi để chuẩn bị
   notebook.

Service (ONNX Runtime, pyvi, bộ phát hiện chủ đề bằng numpy) không cần `pyarrow` hay `onnx`, nên
vẫn chạy được ngay cả khi hai gói này bị chặn.

---

## 11. Chạy H12′ trên Kaggle

**Vì sao.** Từ chiều 2026-10-01, Windows Application Control chặn PyTorch, pyarrow và scikit-learn
trên laptop (mục 9), nên laptop không huấn luyện được. Seed 42 của H12′ đã chạy xong trước đó và
**đủ điều kiện** đi tiếp (UIT-VSFC +0,0104, NEU-ESC +0,0611). Còn lại 4 seed (1337, 2024, 7, 31337)
và nhãn dự đoán của các mô hình đối chứng trên dữ liệu xác nhận.

Notebook: [`notebooks/kaggle_h12p.ipynb`](../notebooks/kaggle_h12p.ipynb). Nó tự tải mã nguồn ở đúng
commit đã ghim từ GitHub, tự tải dữ liệu, và **tự dừng** nếu máy Kaggle thấy dữ liệu khác laptop
(phiên bản pyvi, bộ khôi phục dấu, cách tách tập giữ lại, cách tách ViLexNorm). Bạn không phải sửa
dòng code nào.

**Phần A (bắt buộc): 4 seed, khoảng 1,5–2 giờ GPU.** Các bước:

1. Vào <https://www.kaggle.com> → *Create* → *New Notebook* → *File* → *Import Notebook* → chọn file
   `notebooks/kaggle_h12p.ipynb` trong repo.
2. Thanh bên phải, *Session options*:
   - *Accelerator*: **GPU T4 x2** (hoặc T4; notebook chỉ dùng một GPU).
   - *Internet*: **On** (Kaggle có thể yêu cầu xác minh số điện thoại một lần).
3. *Add-ons* → *Secrets* → *Add a new secret*:
   - tên **`HF_TOKEN`**;
   - giá trị là token Hugging Face của bạn, cái đang nằm trong file `.env`, của tài khoản đã bấm
     đồng ý điều kiện dữ liệu NEU-ESC.

   Bật công tắc gắn secret này vào notebook. **Không dán token vào chat với tôi.**
4. Để notebook ở chế độ **Private** (mặc định).
5. *Save Version* → chọn **Save & Run All (Commit)** → *Save*. Notebook chạy nền tối đa 12 giờ, bạn
   tắt trình duyệt cũng được.
6. Khi phiên chạy xong (trạng thái *Successful*), mở phiên bản đó → tab *Output* → tải
   **`h12p_results.zip`**. File này chỉ có số liệu và nhãn dự đoán, không có câu chữ nào.
7. Chép `h12p_results.zip` vào thư mục repo và báo tôi. Tôi sẽ chạy:
   ```powershell
   & $PY -m vifeedback.cli study h12p-import h12p_results.zip
   & $PY -m vifeedback.cli study h12p-status
   ```

**Phần B (tuỳ chọn, khuyến nghị nếu laptop vẫn bị chặn): khoảng 15 phút GPU.** Phần này tính nhãn
dự đoán của 5 mô hình đối chứng (H10b) và của H12′ seed 42 trên dữ liệu xác nhận. Nó cần 6 thư mục
checkpoint (khoảng 3,1 GB) chỉ có trên laptop. Nếu bạn bỏ qua phần B, tôi làm việc này trên laptop
ngay khi Windows cho chạy PyTorch lại.

1. Nén 6 thư mục (PowerShell, tại thư mục repo):
   ```powershell
   $dirs = Get-ChildItem models -Directory | Where-Object {
       $_.Name -like 'p15-sent-phobert-base-seg_pyvi-h10b-anchored_orig-s*-ckp' -or
       $_.Name -eq 'p16-sent-phobert-base-seg_pyvi-h12p-two_heads_anchored-s42-2842e32b-ckp' }
   $dirs.Name   # phải in ra đúng 6 tên
   Compress-Archive -Path $dirs.FullName -DestinationPath h12p_checkpoints.zip -CompressionLevel Fastest
   ```
2. Kaggle → *Datasets* → *New Dataset*:
   - tải `h12p_checkpoints.zip` lên (Kaggle tự giải nén);
   - đặt tên tuỳ ý, để **Private**, bấm *Create*.
3. Trong notebook: *Add Input* → chọn dataset vừa tạo, rồi làm bước 5 của phần A. Notebook tự tìm
   6 thư mục theo tên. Thiếu thư mục nào thì nó báo và bỏ qua phần B, không hỏng phần A.
4. Sau khi xong, xoá `h12p_checkpoints.zip` trên laptop nếu cần chỗ trống. Thư mục `models\` vẫn
   giữ bản gốc.

**Nếu gặp lỗi:**

| Thông báo | Nghĩa là | Cách xử lý |
|---|---|---|
| `... is gated` hoặc 401/403 ở bước NEU-ESC | Thiếu `HF_TOKEN`, hoặc token của tài khoản chưa đồng ý điều kiện | Kiểm tra secret; mở trang NEU-ESC trên Hugging Face bằng đúng tài khoản đó và bấm đồng ý |
| `pyvi segmentation differs from the laptop` | Kaggle cài được pyvi khác bản 0.1.1 | Báo tôi kèm dòng lỗi |
| `sha256 does not match the manifest` ở bước restorer | Dữ liệu UIT-VSFC tải về khác bản đã ghim | Báo tôi |
| `split differs from ...` | Phiên bản numpy của Kaggle sinh hoán vị khác | Báo tôi; notebook dừng trước khi huấn luyện nên không tốn GPU |
| Hết 12 giờ giữa chừng | Không xảy ra với ước tính 2–2,5 giờ | Chạy lại; seed đã xong sẽ được bỏ qua trong cùng phiên |

---

## Những việc bạn *không* cần làm

- Chạy huấn luyện, đánh giá, commit, cập nhật tài liệu: tôi làm, mỗi lần một job nặng, có thời gian
  nghỉ để máy không quá nóng.
- Chạy Kaggle: Cycle 5 (b) và (a) vừa GPU của laptop. Chỉ khi GPU laptop bận thì mới cần Kaggle, và
  lúc đó tôi sẽ nói rõ notebook nào.
- Dán API key hay token vào chat: chúng nằm trong `.env`.
