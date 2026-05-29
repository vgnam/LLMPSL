# Tóm tắt phương pháp LLM-PSL

## 1. Mục tiêu

LLM-PSL, viết tắt của **LLM-based Pareto Set Learning**, là một phương pháp dùng mô hình ngôn ngữ lớn để tự động sinh heuristic cho các bài toán tối ưu đa mục tiêu. Thay vì chỉ tìm một heuristic tốt nhất, phương pháp hướng tới việc học một tập heuristic đại diện cho nhiều đánh đổi khác nhau, ví dụ:

- Heuristic cho chất lượng lời giải cao nhưng chạy chậm hơn.
- Heuristic cân bằng giữa chất lượng và thời gian chạy.
- Heuristic chạy nhanh nhưng chấp nhận chất lượng thấp hơn.
- Heuristic có cấu trúc mã mới, giúp mở rộng hướng tìm kiếm.

Điểm chính của LLM-PSL là kết hợp **Pareto Set Learning** với sinh chương trình bằng LLM. Mỗi vùng đánh đổi được biểu diễn bằng một vector ưu tiên `lambda`, và hệ thống duy trì bộ nhớ các heuristic phù hợp với từng `lambda`.

## 2. Bài toán tối ưu

Mỗi heuristic sinh ra được đánh giá bằng ba mục tiêu tối thiểu hóa:

```text
F(h) = (q(h), t(h), d(h))
```

Trong đó:

```text
q(h) = -HV(h)
t(h) = runtime(h)
d(h) = -Novelty_CodeBLEU(h)
```

Do đó, điểm số lưu trong quần thể là:

```text
score(h) = (-HV(h), runtime(h), -CodeBLEU_novelty(h))
```

Ý nghĩa:

- `-HV` càng nhỏ thì hypervolume thật càng lớn, tức chất lượng lời giải càng tốt.
- `runtime` càng nhỏ thì heuristic càng nhanh.
- `-CodeBLEU_novelty` càng nhỏ thì mã nguồn càng mới lạ so với các heuristic đã có.

Phương pháp dùng `HV` làm giá trị chất lượng ổn định. `HVI` chỉ phù hợp làm tín hiệu phụ hoặc tiêu chí tie-break vì nó phụ thuộc vào archive hiện tại.

## 3. Vector ưu tiên `lambda`

LLM-PSL dùng vector ưu tiên:

```text
lambda = (lambda_q, lambda_t, lambda_n)
```

với điều kiện:

```text
lambda_i >= 0
sum_i lambda_i = 1
```

Trong đó:

- `lambda_q`: mức ưu tiên cho chất lượng lời giải, tương ứng mục tiêu `-HV`.
- `lambda_t`: mức ưu tiên cho thời gian chạy.
- `lambda_n`: mức ưu tiên cho độ mới của mã nguồn.

Ví dụ:

```text
(0.80, 0.10, 0.10) -> ưu tiên chất lượng
(0.45, 0.45, 0.10) -> cân bằng chất lượng và tốc độ
(0.20, 0.70, 0.10) -> ưu tiên tốc độ
(0.35, 0.25, 0.40) -> ưu tiên novelty của mã
```

Trong triển khai hiện tại, các vector ưu tiên được lấy từ một lưới hữu hạn. Mỗi vector tương ứng với một vùng đánh đổi trên Pareto set.

## 4. Bộ nhớ Pareto theo preference

LLM-PSL duy trì hai cấu trúc chính:

```text
A: archive/quần thể Pareto toàn cục
M: bộ nhớ theo preference
```

Archive `A` lưu các heuristic không bị trội theo ba mục tiêu:

```text
(-HV, runtime, -CodeBLEU_novelty)
```

Bộ nhớ `M` ánh xạ mỗi vector ưu tiên tới một tập nhỏ heuristic đại diện:

```text
M[lambda_k] = {h_1, h_2, ..., h_K}
```

Ý nghĩa của ánh xạ này là:

```text
lambda_k -> tập heuristic đại diện cho vùng đánh đổi lambda_k
```

Đây là khác biệt quan trọng so với cách chỉ duy trì một Pareto population. LLM-PSL học một ánh xạ rời rạc từ preference sang heuristic, gần với tinh thần của Pareto Set Learning.

## 5. CodeBLEU novelty

Để khuyến khích đa dạng mã nguồn, LLM-PSL dùng độ tương đồng CodeBLEU giữa hai chương trình. Vì CodeBLEU có thể bất đối xứng, phương pháp dùng dạng đối xứng:

```text
sim(a, b) = 0.5 * (CodeBLEU(a, b) + CodeBLEU(b, a))
dist(a, b) = 1 - sim(a, b)
```

Novelty của một heuristic được tính bằng khoảng cách trung bình tới `k` chương trình gần nhất:

```text
Novelty(h) = mean_k_smallest { 1 - sim(h, g) | g in archive }
```

Nếu chưa có archive, novelty được khởi tạo bằng `1.0`. Trong code hiện tại, nếu thư viện CodeBLEU không có sẵn, hệ thống dùng fallback dựa trên token và AST similarity để vẫn chạy được.

## 6. Smooth Tchebycheff Set Scalarization

Để cập nhật bộ nhớ `M[lambda]`, LLM-PSL không chọn một heuristic đơn lẻ mà chọn một tập nhỏ các heuristic bổ sung cho nhau. Đây là điểm khác với scalarization thông thường: một cá thể không cần phải tốt ở cả ba mục tiêu, miễn là cả tập có cá thể phù hợp cho từng mục tiêu.

Trước hết, với quần thể hiện tại, hệ thống chỉ xét các heuristic đã có `score`. Mỗi score có dạng:

```text
score(h) = (f_1(h), f_2(h), f_3(h))
         = (-HV(h), runtime(h), -novelty(h))
```

Tất cả đều là mục tiêu cần minimize. Với từng mục tiêu `i`, code tính:

```text
ideal_i = min_h f_i(h)
nadir_i = max_h f_i(h)
```

Sau đó chuẩn hóa score của mỗi heuristic:

```text
s_i(h) = (f_i(h) - ideal_i) / (nadir_i - ideal_i + eps)
```

Sau chuẩn hóa:

- `s_i(h) = 0` nghĩa là heuristic đó đang tốt nhất quần thể ở mục tiêu `i`.
- `s_i(h) = 1` nghĩa là heuristic đó đang kém nhất quần thể ở mục tiêu `i`.
- Giá trị càng nhỏ càng tốt.

Với một tập heuristic `H_K`, Tchebycheff Set Scalarization xét mỗi mục tiêu theo cá thể tốt nhất trong tập:

```text
g_TCH-Set(H_K | lambda) =
    max_i lambda_i * min_{h in H_K} s_i(h)
    + rho * sum_i lambda_i * min_{h in H_K} s_i(h)
```

Trong công thức trên, phần:

```text
min_{h in H_K} s_i(h)
```

có nghĩa là với mỗi mục tiêu `i`, tập `H_K` lấy cá thể tốt nhất của riêng mục tiêu đó. Vì vậy một tập có thể gồm:

- Một heuristic rất tốt về `-HV`.
- Một heuristic rất tốt về `runtime`.
- Một heuristic rất tốt về `novelty`.

Tập này có thể tốt hơn việc bắt một heuristic duy nhất phải cân bằng tất cả mục tiêu.

Triển khai hiện tại dùng phiên bản smooth với log-sum-exp để xếp hạng ổn định hơn. Với mỗi mục tiêu, `min` được thay bằng soft-min:

```text
r_i(H_K) = softmin_{h in H_K} s_i(h)
         = -mu * log sum_{h in H_K} exp(-s_i(h) / mu)
```

Sau đó `max` giữa các mục tiêu được thay bằng soft-max:

```text
g_STCH-Set(H_K | lambda) =
    mu * log sum_i exp(lambda_i * r_i(H_K) / mu)
    + rho * sum_i lambda_i * r_i(H_K)
```

Trong đó:

- `lambda_i` càng lớn thì mục tiêu `i` càng quan trọng.
- `mu = smooth_mu`, hiện tại là `0.05`, điều khiển độ mượt của soft-min/soft-max.
- `rho = tchebycheff_rho`, hiện tại là `0.05`, là phần phạt phụ để tránh nghiệm lệch quá mạnh.
- Giá trị `g_STCH-Set` càng nhỏ thì tập heuristic càng phù hợp với preference đó.

### Cách chọn archive/anchor cụ thể

Trong báo cáo có thể gọi `A` là archive, nhưng trong code hiện tại tập được dùng trực tiếp để chọn là `_population`: quần thể đã được lọc bằng non-dominated sorting, crowding distance và code diversity. Vì vậy, khi nói “chọn archive bằng Smooth Tchebycheff”, cần hiểu chính xác là:

```text
Smooth Tchebycheff Set chọn một subset đại diện từ population/archive hiện tại
để lưu vào M[lambda] và dùng làm anchor cho lần sinh tiếp theo.
```

Với mỗi preference `lambda_k`, hàm `update_preference_memory()` làm các bước sau:

```text
Input:
  P = current population/archive, chỉ gồm các heuristic đã có score
  Lambda = {lambda_1, ..., lambda_K}
  set_size = tchebycheff_set_size, hiện tại bằng 3

For each lambda_k in Lambda:
  1. Tính ideal và nadir trên toàn bộ P:
       ideal_i = min_{h in P} f_i(h)
       nadir_i = max_{h in P} f_i(h)

  2. Gọi greedy_tchebycheff_set_selection:
       selected_set = argmin xấp xỉ g_STCH-Set(H | lambda_k)
       với H là subset của P và |H| <= set_size

  3. Tính lại giá trị của selected_set:
       value_k = g_STCH-Set(selected_set | lambda_k)

  4. Lưu vào memory:
       M[lambda_k]["set"] = selected_set
       M[lambda_k]["value"] = value_k
       M[lambda_k]["updated"] = generation hiện tại
```

Vì duyệt tất cả subset là tốn kém, `greedy_tchebycheff_set_selection` chọn greedy từng cá thể một:

```text
Input:
  P = population/archive hiện tại
  lambda = preference đang xét
  K = tchebycheff_set_size, hiện tại K = 3

Initialize:
  selected = empty
  candidates = P

While len(selected) < K:
  For each candidate h in candidates:
      trial_set = selected + [h]
      value(h) = g_STCH-Set(trial_set | lambda)

  Chọn h_best có value(h) nhỏ nhất
  selected = selected + [h_best]
  candidates = candidates - {h_best}

Return selected
```

Điểm quan trọng: cá thể được chọn ở mỗi bước không nhất thiết là cá thể tốt nhất riêng lẻ theo tất cả mục tiêu. Nó là cá thể làm cho **cả tập hiện tại** có giá trị `g_STCH-Set` nhỏ nhất.

Cụ thể hơn, giả sử đã chọn được:

```text
selected = {h_a}
```

Khi xét một candidate mới `h_b`, hệ thống không hỏi:

```text
h_b có score riêng lẻ tốt nhất không?
```

mà hỏi:

```text
selected + {h_b} có làm g_STCH-Set của cả tập giảm nhiều nhất không?
```

Vì vậy `h_b` có thể được chọn nếu nó bổ sung phần yếu của `h_a`. Ví dụ `h_a` rất tốt về `-HV` nhưng chạy chậm; một `h_b` có runtime rất thấp có thể làm tập `{h_a, h_b}` phù hợp hơn với preference cân bằng hoặc preference thiên về runtime.

Ví dụ với `lambda = (0.80, 0.10, 0.10)`, mục tiêu `-HV` quan trọng nhất. Greedy thường sẽ chọn trước heuristic có `-HV` tốt. Nhưng ở bước sau, nếu tập đã có một heuristic rất mạnh về `-HV`, thuật toán có thể chọn thêm heuristic chạy nhanh hơn hoặc mới lạ hơn nếu cá thể đó làm giảm phần yếu còn lại của tập.

Ví dụ với `lambda = (0.20, 0.70, 0.10)`, runtime có trọng số lớn nhất. Greedy sẽ ưu tiên tập có ít nhất một heuristic rất nhanh. Sau đó các cá thể tiếp theo được thêm vào nếu chúng cải thiện thêm chất lượng hoặc novelty mà không làm score set xấu đi theo preference.

Sau khi memory đã có `M[lambda]["set"]`, hàm `selection(selection_num, preference)` dùng memory này để lấy anchor:

```text
Input:
  lambda_t = preference mục tiêu
  selection_num = số parent cần lấy

1. Chuẩn hóa lambda_t để tổng trọng số bằng 1.

2. Tìm memory cell tương ứng:
     entry = M[lambda_t]

3. Nếu entry có đủ anchor:
     return entry["set"][:selection_num]

4. Nếu entry chưa đủ anchor:
     selected = các anchor đang có trong M[lambda_t]
     thiếu bao nhiêu thì lấy thêm từ population bằng parent_selection(...)

5. parent_selection lại dùng greedy_tchebycheff_set_selection
   để chọn phần còn thiếu theo cùng lambda_t.

Return:
  danh sách anchor/parent dùng cho prompt LLM
```

Như vậy Smooth Tchebycheff Set xuất hiện ở hai chỗ liên quan đến archive/anchor:

```text
1. Cập nhật M[lambda]:
     chọn subset tốt nhất từ population/archive cho từng lambda.

2. Khi cần parent:
     lấy từ M[lambda_t]; nếu thiếu thì dùng lại Smooth Tchebycheff Set
     để fill thêm parent từ population.
```

Khi cập nhật memory, với mỗi `lambda_k`, hệ thống lưu:

```text
M[lambda_k]["set"]      = selected_set
M[lambda_k]["function"] = selected_set[0]
M[lambda_k]["value"]    = g_STCH-Set(selected_set | lambda_k)
M[lambda_k]["updated"]  = generation hiện tại
```

Nếu `selected_set` thay đổi hoặc giá trị scalarization tốt hơn giá trị cũ, memory cell đó được cập nhật.

### Ví dụ ngắn về chọn archive

Giả sử archive hiện có năm heuristic:

```text
h1: chất lượng rất tốt, chạy chậm, novelty trung bình
h2: chất lượng khá, chạy rất nhanh, novelty thấp
h3: chất lượng trung bình, runtime trung bình, novelty rất cao
h4: cân bằng cả ba mục tiêu
h5: chất lượng kém, chạy nhanh, novelty cao
```

Với `lambda = (0.80, 0.10, 0.10)`, Smooth Tchebycheff Set có thể chọn:

```text
M[(0.80, 0.10, 0.10)] = {h1, h4, h3}
```

vì `h1` bao phủ tốt mục tiêu chất lượng, còn `h4/h3` bổ sung runtime hoặc novelty.

Với `lambda = (0.10, 0.80, 0.10)`, nó có thể chọn:

```text
M[(0.10, 0.80, 0.10)] = {h2, h5, h4}
```

vì preference này cần ít nhất một heuristic chạy rất nhanh.

Với `lambda = (0.30, 0.30, 0.40)`, nó có thể chọn:

```text
M[(0.30, 0.30, 0.40)] = {h3, h4, h2}
```

vì novelty có trọng số lớn, nhưng vẫn cần giữ chất lượng và runtime không quá kém.

## 7. Bộ lập lịch chọn preference

Ở mỗi vòng lặp, LLM-PSL chọn một preference mục tiêu:

```text
lambda_t = SelectPreference(M, A)
```

Preference không được chọn ngẫu nhiên hoàn toàn. Code hiện tại chọn theo trạng thái của bộ nhớ `M`.

Lưới preference mặc định cho ba mục tiêu gồm:

```text
(0.80, 0.10, 0.10) -> ưu tiên chất lượng
(0.65, 0.25, 0.10) -> thiên về chất lượng
(0.45, 0.45, 0.10) -> cân bằng chất lượng và runtime
(0.25, 0.65, 0.10) -> thiên về runtime
(0.10, 0.80, 0.10) -> ưu tiên runtime
(0.50, 0.20, 0.30) -> chất lượng + novelty
(0.30, 0.30, 0.40) -> novelty cao, vẫn cân bằng hai mục tiêu còn lại
(0.20, 0.40, 0.40) -> runtime + novelty
(1/3, 1/3, 1/3)     -> cân bằng cả ba mục tiêu
```

Smooth Tchebycheff Set không trực tiếp random chọn preference. Nó tạo ra `value_k` cho từng memory cell:

```text
value_k = g_STCH-Set(M[lambda_k]["set"] | lambda_k)
```

`value_k` là mức độ bao phủ của archive/anchor hiện tại đối với preference `lambda_k`. Vì đây là bài toán minimize:

```text
value_k thấp -> vùng lambda_k đang được bao phủ tốt
value_k cao  -> vùng lambda_k còn yếu, cần sinh thêm heuristic
```

Quy trình chọn preference trong `select_target_preference`:

```text
1. Tìm các preference có memory rỗng:
      empty = {lambda_k | M[lambda_k]["set"] is empty}

2. Nếu còn preference rỗng:
      chọn ngẫu nhiên một lambda trong empty

3. Nếu tất cả preference đã có đại diện:
      với mỗi lambda_k:
          value_k = M[lambda_k]["value"]
          stale_k = generation hiện tại - M[lambda_k]["updated"]

      chuẩn hóa value_k về [0, 1]

      priority(lambda_k) =
          normalized_value_k
          + 0.05 * stale_k
          + random_noise

      chọn lambda có priority lớn nhất
```

Trong đó:

- `normalized_value_k` đến từ Smooth Tchebycheff Set. Nó đo vùng preference nào đang có set đại diện kém hơn.
- `stale_k = generation - M[lambda_k]["updated"]`. Nó đo vùng nào lâu chưa được cập nhật.
- `random_noise` rất nhỏ, khoảng `0.01`, dùng để phá hòa.

`stale_k` giúp tránh việc một vùng bị bỏ quên quá lâu. Nếu một preference lâu không được cập nhật, điểm ưu tiên của nó tăng dần theo số generation.

Ví dụ:

```text
lambda_A:
  value = 0.10, stale = 0
  -> vùng này đang tốt, không cần ưu tiên

lambda_B:
  value = 0.75, stale = 1
  -> vùng này yếu, nên ưu tiên

lambda_C:
  value = 0.40, stale = 8
  -> không quá yếu nhưng đã lâu chưa cập nhật, cũng có thể được chọn
```

Nếu bỏ qua noise, priority sẽ có dạng:

```text
priority_A = norm(0.10) + 0.05 * 0
priority_B = norm(0.75) + 0.05 * 1
priority_C = norm(0.40) + 0.05 * 8
```

Như vậy scheduler cân bằng hai mục tiêu:

```text
khai thác vùng yếu theo Smooth Tchebycheff value
và quay lại vùng bị stale quá lâu
```

Tóm lại, preference được ưu tiên khi:

- Chưa có heuristic đại diện.
- Lâu chưa được cải thiện.
- Giá trị scalarization còn kém.

Cách này giúp phân bổ ngân sách đánh giá vào các vùng Pareto còn yếu hoặc chưa được bao phủ. Sau khi chọn `lambda_t`, hệ thống dùng `M[lambda_t]["set"]` làm anchor chính để sinh heuristic mới bằng LLM. Nếu số anchor trong memory chưa đủ, hệ thống lấy thêm cá thể từ quần thể bằng cùng cơ chế Tchebycheff Set selection.

Có thể tóm tắt quan hệ giữa Smooth Tchebycheff, archive và preference như sau:

```text
Population/archive hiện tại
  -> dùng Smooth Tchebycheff Set để chọn M[lambda_k]["set"] cho từng lambda_k
  -> mỗi M[lambda_k] có một value_k
  -> scheduler chọn lambda_t có value/staleness cao
  -> lấy M[lambda_t]["set"] làm anchor
  -> LLM sinh heuristic mới
  -> evaluate thật
  -> cập nhật population/archive
  -> cập nhật lại M bằng Smooth Tchebycheff Set
```

## 8. Sinh heuristic bằng LLM

Sau khi chọn `lambda_t`, hệ thống chọn các heuristic anchor từ bộ nhớ và archive. Các anchor đại diện cho những cơ chế có ích, ví dụ chất lượng tốt, chạy nhanh hoặc có mã khác biệt.

Điểm cần phân biệt rõ:

```text
Smooth Tchebycheff Set không trực tiếp chọn toán tử LLM.
Smooth Tchebycheff Set dùng để chọn preference, chọn anchor/parent và cập nhật memory.
Các toán tử LLM được chạy theo lịch trong vòng lặp sinh mẫu và các flag cấu hình.
```

Quy trình đúng là:

```text
1. select_target_preference()
      -> chọn lambda_t

2. population.selection(..., preference=lambda_t)
      -> chọn anchor/parent phù hợp với lambda_t
      -> ưu tiên lấy từ M[lambda_t]["set"]
      -> nếu thiếu thì dùng Tchebycheff Set selection để lấy thêm từ population

3. Chạy các prompt/operator LLM đang bật
      -> candidate chính: interpolation hoặc E1 fallback
      -> candidate phụ: E2, M1, M2 nếu các flag tương ứng bật

4. LLM sinh code

5. Nếu bật novelty repair và code quá giống chương trình cũ
      -> gọi novelty repair

6. Đánh giá thật bằng evaluator
```

### Khi nào dùng từng toán tử

Nói chung, LLM-PSL sẽ cố gắng dùng **tất cả toán tử đang active**, không phải chọn một toán tử duy nhất cho cả vòng lặp. Trong cấu hình hiện tại, `main.py` bật `use_psl_interpolation=True`, còn `use_e2_operator`, `use_m1_operator`, `use_m2_operator` được kế thừa mặc định là `True` từ lớp nền. Vì vậy sau khởi tạo, một vòng lặp có thể lần lượt sinh candidate bằng:

```text
Interpolation/E1 -> E2 -> M1 -> M2
```

Mỗi lần sinh candidate, hệ thống lại chọn `lambda_t` và parent/anchor theo preference tương ứng. Nếu đã đạt `max_sample_nums` thì vòng lặp dừng, nên không phải lúc nào cả bốn candidate đều được sinh đủ trong vòng cuối.

### Pareto interpolation

Đây là toán tử chính trong cấu hình hiện tại.

Điều kiện dùng trong code:

```text
if use_psl_interpolation == True and số anchor >= 2:
    dùng get_prompt_interpolate(...)
else:
    dùng get_prompt_e1(...)
```

Với cấu hình trong `main.py`:

```text
use_psl_interpolation = True
selection_num mặc định = 2
```

nên sau giai đoạn khởi tạo, nếu hệ thống chọn được ít nhất hai anchor, nó sẽ ưu tiên Pareto interpolation.

Ý nghĩa:

```text
h_t = LLM_Interpolate(h_a, h_b, lambda_t)
```

LLM nhận nhiều anchor cùng score của chúng và target preference `lambda_t`. Prompt yêu cầu kết hợp cơ chế tốt từ các parent:

```text
- Giữ cơ chế giúp tăng HV từ heuristic mạnh về chất lượng.
- Giữ cơ chế giúp giảm runtime từ heuristic chạy nhanh.
- Tạo thay đổi thuật toán thật sự để tăng novelty.
```

Ví dụ: với `lambda_t = (0.80, 0.10, 0.10)`, prompt sẽ thiên về cơ chế cải thiện hypervolume. Với `lambda_t = (0.10, 0.80, 0.10)`, prompt sẽ thiên về logic đơn giản và chạy nhanh hơn.

### E1 fallback

Nếu không đủ điều kiện dùng interpolation, hệ thống dùng `get_prompt_e1`.

Điều kiện:

```text
if use_psl_interpolation == False or số anchor < 2:
    dùng get_prompt_e1(...)
```

E1 vẫn là toán tử kết hợp/crossover từ các parent, nhưng prompt không gọi rõ là Pareto interpolation. Nó yêu cầu LLM phân tích các thuật toán hiện có, kết hợp cơ chế hữu ích và tạo một heuristic mới theo `lambda_t`.

### E2

Nếu `use_e2_operator = True`, sau candidate chính, vòng lặp có thể sinh thêm một candidate bằng `get_prompt_e2`.

Điều kiện:

```text
if use_e2_operator == True:
    chọn lại lambda_t
    chọn lại anchor theo lambda_t
    dùng get_prompt_e2(...)
```

E2 cũng dùng nhiều parent, nhưng prompt nhấn mạnh việc tìm backbone chung hoặc ý tưởng chính từ các parent rồi tạo dạng thuật toán khác.

### M1

Nếu `use_m1_operator = True`, hệ thống chọn một parent và yêu cầu LLM sửa đổi heuristic đó.

Điều kiện:

```text
if use_m1_operator == True:
    chọn lambda_t
    chọn 1 parent theo lambda_t
    dùng get_prompt_m1(...)
```

M1 phù hợp cho khai thác cục bộ quanh một heuristic đang có. Prompt yêu cầu thay đổi một trong các thành phần:

```text
selection policy
neighborhood move
repair logic
scoring formula
```

### M2

Nếu `use_m2_operator = True`, hệ thống cũng chọn một parent nhưng tập trung vào thay đổi tham số hoặc ngưỡng trong thuật toán.

Điều kiện:

```text
if use_m2_operator == True:
    chọn lambda_t
    chọn 1 parent theo lambda_t
    dùng get_prompt_m2(...)
```

M2 phù hợp khi heuristic đã có cấu trúc tốt nhưng cần điều chỉnh mức độ tham lam, xác suất random, threshold chọn nghiệm hoặc tiêu chí ưu tiên để hợp với `lambda_t`.

### Pareto extrapolation

Trong thiết kế phương pháp, Pareto extrapolation là toán tử đi xa hơn từ một anchor theo một hướng preference:

```text
h_t = LLM_Extrapolate(h_a, lambda_t)
```

Ví dụ: đẩy heuristic cân bằng sang hướng chất lượng cao hơn, hoặc đẩy heuristic đang chạy nhanh sang hướng còn nhanh hơn.

Tuy nhiên, trong code hiện tại, `get_prompt_extrapolate` đã được định nghĩa trong `prompt.py` nhưng vòng lặp chính trong `eoh.py` chưa gọi trực tiếp toán tử này. Vai trò gần nhất với extrapolation hiện đang được thực hiện bởi `M1` và `M2`, vì hai toán tử này chọn một parent rồi sửa nó theo target preference.

### Novelty repair

Novelty repair không phải toán tử sinh chính ban đầu. Nó là bước hậu xử lý sau khi LLM đã sinh một candidate.

Điều kiện:

```text
if use_novelty_repair == True
and candidate gần trùng với chương trình cũ theo CodeBLEU/fallback similarity:
    dùng get_prompt_novelty_repair(...)
```

Ngưỡng gần trùng lặp hiện tại:

```text
novelty_threshold = 0.9
```

Nếu similarity giữa candidate và một chương trình cũ lớn hơn hoặc bằng ngưỡng này, candidate bị xem là quá giống. Khi đó prompt yêu cầu LLM viết lại bằng cơ chế thuật toán khác, chẳng hạn đổi chính sách chọn nghiệm, toán tử lân cận, repair logic hoặc công thức scoring.

Trong `main.py` hiện tại:

```text
use_novelty_repair = False
```

nên novelty repair đang tắt mặc định.

### Tóm tắt chọn toán tử

```text
Khởi tạo population:
  dùng I1 để sinh heuristic ban đầu theo các lambda khác nhau.

Vòng lặp chính:
  luôn chọn lambda_t trước.
  luôn chọn anchor/parent theo lambda_t.

  Candidate chính:
    nếu use_psl_interpolation và có >= 2 anchor:
        dùng Pareto interpolation
    ngược lại:
        dùng E1

  Sau candidate chính, tiếp tục dùng các toán tử phụ nếu flag bật:
    use_e2_operator -> dùng E2 với nhiều parent
    use_m1_operator -> dùng M1 với một parent
    use_m2_operator -> dùng M2 với một parent

  Với cấu hình hiện tại:
    use_psl_interpolation = True
    use_e2_operator = True
    use_m1_operator = True
    use_m2_operator = True

  Vì vậy hệ thống sẽ cố gắng dùng lần lượt:
    Interpolation/E1, rồi E2, rồi M1, rồi M2

  Sau khi sinh candidate:
    nếu use_novelty_repair và candidate quá giống code cũ:
        dùng novelty repair

Smooth Tchebycheff Set:
  không chọn operator
  chỉ chọn lambda, chọn anchor/parent, và cập nhật M[lambda]

Lưu ý:
  get_prompt_extrapolate đã có trong prompt.py nhưng vòng lặp chính hiện chưa gọi trực tiếp.
  use_novelty_repair hiện đang False trong main.py, nên repair tắt mặc định.
```

## 9. Luồng thuật toán

Quy trình tổng quát:

```text
Input:
  task description
  template function
  preference grid Lambda
  max evaluations T
  population size N

Initialize:
  A = empty archive
  M[lambda_k] = empty for each lambda_k in Lambda

Initial sampling:
  sinh heuristic ban đầu theo các preference khác nhau
  đánh giá bằng evaluator thật
  tính novelty bằng CodeBLEU
  cập nhật archive và preference memory

For t = 1 ... T:
  1. Chọn preference mục tiêu lambda_t
  2. Chọn anchor từ M[lambda_t] và archive
  3. Sinh heuristic mới bằng LLM interpolation, E1/E2 hoặc M1/M2
  4. Kiểm tra cú pháp, chữ ký hàm, duplicate và near-duplicate
  5. Đánh giá thật để lấy HV và runtime
  6. Tính CodeBLEU novelty
  7. Gán score = (-HV, runtime, -novelty)
  8. Cập nhật Pareto archive và bộ nhớ M

Return:
  archive Pareto toàn cục A
  preference-indexed memory M
```

## 10. Khác biệt so với MPaGE/LLM-PFG

MPaGE/LLM-PFG chủ yếu duy trì Pareto population và dùng Pareto front grid để chọn parent. Trong khi đó, LLM-PSL bổ sung:

- Bộ nhớ theo preference `M[lambda]`.
- Lưới preference để bao phủ nhiều vùng đánh đổi.
- Smooth Tchebycheff Set Scalarization để chọn tập heuristic đại diện.
- CodeBLEU novelty như mục tiêu thứ ba.
- LLM Pareto interpolation/extrapolation trong không gian chương trình.
- Cơ chế chọn preference thích nghi để tập trung vào vùng còn yếu.

Vì vậy, đóng góp không chỉ nằm ở prompt có thêm `lambda`, mà nằm ở toàn bộ cơ chế tìm kiếm: trạng thái bộ nhớ, chính sách chọn parent, cập nhật archive và toán tử sinh chương trình đều phụ thuộc vào preference.

## 11. Tham số triển khai hiện tại

Trong `main.py`, LLM-PSL được cấu hình với các tham số chính:

```text
objective_num = 3
novelty_k = 8
novelty_threshold = 0.9
prefer_codebleu = True
tchebycheff_rho = 0.05
tchebycheff_set_size = 3
smooth_set_scalarization = True
smooth_mu = 0.05
use_psl_interpolation = True
use_novelty_repair = False
```

Ý nghĩa:

- `objective_num = 3`: dùng ba mục tiêu `-HV`, `runtime`, `-novelty`.
- `novelty_k`: số láng giềng gần nhất dùng để tính novelty.
- `novelty_threshold`: ngưỡng phát hiện chương trình gần trùng lặp.
- `tchebycheff_set_size = 3`: mỗi preference lưu một tập nhỏ gồm tối đa ba heuristic.
- `smooth_set_scalarization = True`: dùng phiên bản smooth của Tchebycheff Set.
- `use_psl_interpolation = True`: ưu tiên toán tử Pareto interpolation khi có đủ anchor.

## 12. Kết luận ngắn

LLM-PSL mở rộng Pareto Set Learning sang bài toán thiết kế heuristic bằng LLM. Phương pháp thay mô hình Pareto liên tục bằng bộ nhớ rời rạc theo preference, trong đó mỗi `lambda` ánh xạ tới một tập heuristic đại diện. Nhờ kết hợp đánh giá thật bằng hypervolume/runtime, novelty dựa trên CodeBLEU, Smooth Tchebycheff Set Scalarization và các toán tử sinh chương trình có điều kiện theo preference, LLM-PSL hướng tới việc tạo ra một tập heuristic đa dạng, bao phủ tốt nhiều vùng đánh đổi của bài toán tối ưu đa mục tiêu.
