# Hướng dẫn các việc chủ dự án làm tay

**Cập nhật:** 2026-09-29, sau Cycle 4. Tài liệu này dành cho chủ dự án. Nó liệt kê những việc máy
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
| 1 | [Neutral audit](#1-neutral-audit-gán-nhãn-kiểm-tra-lớp-neutral) | 6–8 giờ, chia nhiều buổi | Càng sớm càng tốt | S9 (≥ 30 lỗi được mã hoá), nhánh D2 cho lớp neutral (S4) |
| 2 | [Duyệt 15 nhãn của challenge v1 (U2)](#2-duyệt-15-nhãn-của-challenge-v1-u2) | 20 phút | Bất kỳ lúc nào | Báo cáo challenge v1 kèm nhãn đã được người kiểm |
| 3 | ~~Chuẩn bị máy rảnh để đo độ trễ~~ | — | ✅ Xong 2026-09-30 (bạn cho phép không cần máy rảnh) | p95 12,3 ms (mô hình nhỏ) |
| 4 | [Quyết định giấy phép nếu mô hình nhỏ (H11) được phát hành](#4-quyết-định-giấy-phép-khi-mô-hình-học-từ-vilexnorm) | 5 phút | Trước khi H11 được phát hành | Bản phát hành mô hình nhỏ |
| 5 | [Duyệt upload lên Hugging Face](#5-duyệt-upload-lên-hugging-face) | 10 phút mỗi lần | Mỗi khi có bản phát hành mới | Bản trên Hub khớp với service |
| 6 | [Quyết định chính sách nhãn cho trường khác](#6-quyết-định-chính-sách-nhãn-cho-văn-bản-của-trường-khác-quyết-định-số-2) | 15 phút đọc | Trước Cycle 5 (c) | Hướng (c) |
| 7 | [Dữ liệu có nhãn mới cho hướng (c)](#7-dữ-liệu-có-nhãn-mới-cho-hướng-c) | 20–25 giờ (hai người) | Trước Cycle 5 (c) | Mọi kết luận mới về văn bản của trường khác |
| 8 | [Challenge v2](#8-tuỳ-chọn-challenge-v2) | 4–6 giờ | Tuỳ chọn | Kiểm tra teencode do người thật gõ |
| 10 | [Chọn mô hình cho service](#10-chọn-mô-hình-cho-service-mô-hình-nhỏ-hay-mô-hình-12-lớp) | 5 phút | Khi tiện | Service dùng mô hình nào; có công bố bản nhỏ hay không |

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
  (transfer set), và mô hình thầy gán nhãn mềm cho chúng. Vì vậy câu hỏi dưới đây áp dụng cho H11.

Việc trọng số học từ dữ liệu SA có phải là "tác phẩm phái sinh" hay không chưa được pháp lý làm rõ.
Cách an toàn là: **nếu mô hình H11 được phát hành, trọng số mang CC BY-NC-SA 4.0**. Thay đổi này không
ảnh hưởng mục đích nghiên cứu phi thương mại của bạn.

**Bạn cần trả lời một câu** trước khi phát hành H11: đồng ý đổi giấy phép trọng số sang
CC BY-NC-SA 4.0 (khuyến nghị), hay giữ CC BY-NC 4.0 và **không** đưa H11 lên Hub (service vẫn dùng
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

---

## 8. (Tuỳ chọn) Challenge v2

Challenge v2 là một bộ câu do **người thật gõ** (teencode, không dấu, viết tắt) với nhãn do người gán,
để kiểm tra điều NEU-ESC không kiểm được: teencode tự nhiên trong đúng văn phong khảo sát môn học.
Nó không chặn việc nào; NEU-ESC đã thay nó làm tập xác nhận (ADR-030). Nếu sau này có người giúp, quy
trình nằm trong [EVALUATION_DATA.md](EVALUATION_DATA.md).

---

## 10. Chọn mô hình cho service: mô hình nhỏ hay mô hình 12 lớp

**Hiện trạng (2026-09-30).**
- Theo quy tắc đã khai báo trước (`cycle5.yaml` v3), service đã chuyển sang **mô hình nhỏ 6 lớp**
  (H11, ADR-039/040).
- Mô hình 12 lớp vẫn còn nguyên ở `models/serve/.previous-sentiment`, và vẫn đang có trên Hugging
  Face.

| | Mô hình nhỏ (đang phục vụ) | Mô hình 12 lớp |
|---|---|---|
| Kích thước | **185 MB** | 540 MB |
| Độ trễ p95 (laptop) | **12,3 ms** | 22,1 ms |
| Macro-F1 trên test (5 seed) | 0,817 | **0,830** |
| Neutral F1 trên test (seed 42) | 0,545 (dưới mức tối thiểu S4 là 0,55) | **0,592** |

Trên validation hai mô hình ngang nhau. Trên test, mô hình nhỏ kém ở cả 5 seed. Tập validation chỉ
có 73 câu neutral nên không phát hiện được chênh lệch cỡ này.

**Bạn chọn một trong hai:**
- **Giữ mô hình nhỏ** nếu ưu tiên kích thước và tốc độ. Mô hình 12 lớp đã đạt mục tiêu 30 ms, nên
  lợi ích thật chủ yếu là dung lượng (S7).
- **Quay lại mô hình 12 lớp** nếu ưu tiên độ chính xác và lớp neutral, vốn là trọng tâm của dự án.
  Chạy trong PowerShell tại thư mục repo:
  ```powershell
  Rename-Item models\serve\sentiment sentiment-student
  Rename-Item models\serve\.previous-sentiment sentiment
  ```
  Rồi báo tôi để cập nhật tài liệu. Không có gì bị xoá; đổi ngược lại cũng chỉ cần hai lệnh đổi tên.

**Khuyến nghị của tôi:** quay lại mô hình 12 lớp cho service, vì dự án đặt lớp neutral lên hàng
đầu và mô hình 12 lớp vẫn nhanh dưới 30 ms. Mô hình nhỏ vẫn có giá trị như một bản rút gọn công bố
riêng, cho ai cần chạy trên máy yếu.

**Công bố mô hình nhỏ lên Hugging Face** (tuỳ chọn, cần bạn duyệt): bản chạy thử đã có ở
`models/publish/vifeedback-sentiment-phobert-6l/`. Đọc `README.md` trong đó (model card đã ghi rõ
khoảng chênh trên test), quyết định giấy phép (mục 4), rồi trả lời "duyệt upload bản nhỏ". Bản này
đi vào repo riêng `Datk4/vifeedback-sentiment-phobert-6l`, không đụng tới repo 12 lớp.

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

## Những việc bạn *không* cần làm

- Chạy huấn luyện, đánh giá, commit, cập nhật tài liệu: tôi làm, mỗi lần một job nặng, có thời gian
  nghỉ để máy không quá nóng.
- Chạy Kaggle: Cycle 5 (b) và (a) vừa GPU của laptop. Chỉ khi GPU laptop bận thì mới cần Kaggle, và
  lúc đó tôi sẽ nói rõ notebook nào.
- Dán API key hay token vào chat: chúng nằm trong `.env`.
