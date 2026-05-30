# Mô tả phương pháp LLM-PSL

LLM-PSL là phương pháp dùng mô hình ngôn ngữ lớn để tự động sinh heuristic cho các bài toán tối ưu đa mục tiêu. Thay vì chỉ tìm một heuristic tốt nhất, phương pháp hướng tới việc tạo ra một tập heuristic đa dạng, mỗi heuristic phù hợp với một kiểu đánh đổi khác nhau giữa chất lượng lời giải, thời gian chạy và độ mới của mã nguồn.

## 1. Mục tiêu

Mỗi heuristic do LLM sinh ra được đánh giá bằng evaluator thật. Score của một heuristic `h` được biểu diễn dưới dạng:

```text
score(h) = (-HV(h), runtime(h), -novelty(h))
```

Trong đó:

- `HV(h)` đo chất lượng lời giải, hypervolume càng lớn càng tốt.
- `runtime(h)` đo thời gian chạy, càng nhỏ càng tốt.
- `novelty(h)` đo độ khác biệt của mã so với các heuristic đã có, càng lớn càng tốt.

Vì thuật toán dùng dạng tối thiểu hóa, nên `HV` và `novelty` được đổi dấu thành `-HV` và `-novelty`.

## 2. Preference và vùng đánh đổi

LLM-PSL chia không gian đánh đổi thành nhiều vùng preference. Mỗi preference là một vector:

```text
lambda = (lambda_quality, lambda_runtime, lambda_novelty)
```

Ví dụ:

```text
(0.80, 0.10, 0.10) -> ưu tiên chất lượng lời giải
(0.45, 0.45, 0.10) -> cân bằng chất lượng và runtime
(0.10, 0.80, 0.10) -> ưu tiên tốc độ
(0.30, 0.30, 0.40) -> ưu tiên độ mới của mã
```

Mỗi `lambda` tương ứng với một vùng đánh đổi trên Pareto set.

## 3. Bộ nhớ theo preference

Điểm chính của LLM-PSL là không chỉ giữ một population chung, mà còn duy trì một bộ nhớ theo preference:

```text
M[lambda] = set heuristic đại diện cho vùng lambda
```

Ví dụ:

```text
M[(0.80, 0.10, 0.10)] = {h1, h4, h7}
M[(0.10, 0.80, 0.10)] = {h2, h5, h8}
```

Các set này được chọn từ population/archive hiện tại. Một heuristic có thể xuất hiện trong nhiều vùng nếu nó phù hợp với nhiều kiểu đánh đổi.

## 4. Chọn set heuristic cho từng preference

Với mỗi `lambda`, hệ thống chọn một set nhỏ các heuristic đại diện, hiện tại có kích thước tối đa:

```text
tchebycheff_set_size = 3
```

Ý tưởng là một set nhiều heuristic có thể bổ sung nhau tốt hơn một heuristic đơn lẻ. Ví dụ:

```text
h1 tốt về chất lượng
h2 tốt về runtime
h3 tốt về novelty
```

thì set `{h1, h2, h3}` có thể là đại diện tốt cho một vùng cân bằng.

Việc chọn set được thực hiện bằng Smooth Tchebycheff Set. Có thể hiểu đơn giản là: với từng candidate, hệ thống thử thêm candidate đó vào set hiện tại, tính xem toàn bộ set có phù hợp với `lambda` hơn không, rồi chọn candidate làm set tốt nhất.

Quy trình:

```text
selected = empty

while selected chưa đủ 3 heuristic:
    thử từng heuristic h trong population
    tính độ phù hợp của selected + {h} với lambda
    chọn h làm set có độ phù hợp tốt nhất
```

Giá trị phù hợp càng nhỏ thì set càng tốt cho preference đó.

## 5. Chọn preference để sinh heuristic mới

Sau khi đã có `M[lambda]` cho các vùng, hệ thống chọn một preference mục tiêu `lambda_t` để cải thiện tiếp.

Nếu có vùng chưa có heuristic đại diện:

```text
chọn một lambda còn rỗng
```

Nếu tất cả vùng đã có đại diện, hệ thống ưu tiên vùng:

- Có set hiện tại còn kém.
- Lâu chưa được cải thiện.
- Cần thêm heuristic mới để mở rộng Pareto set.

Có thể viết đơn giản:

```text
priority(lambda) = mức độ yếu của vùng + độ lâu chưa cập nhật
```

Preference có priority cao nhất sẽ được chọn làm `lambda_t`.

## 6. Sinh heuristic bằng LLM

Sau khi chọn `lambda_t`, hệ thống lấy các heuristic trong:

```text
M[lambda_t]
```

làm anchor/parent cho LLM. Các anchor này được đưa vào prompt cùng với score và target preference. Sau đó LLM sinh heuristic mới bằng các toán tử như:

- Kết hợp nhiều heuristic cha.
- Sửa đổi một heuristic hiện có.
- Điều chỉnh cơ chế chọn nghiệm, toán tử lân cận, scoring hoặc repair.
- Tạo biến thể mới theo hướng preference đang cần cải thiện.

Trong cấu hình hiện tại, hệ thống cố gắng dùng các toán tử đang bật theo thứ tự:

```text
Interpolation/E1 -> E2 -> M1 -> M2
```

Smooth Tchebycheff Set không chọn toán tử LLM. Nó chỉ chọn preference và anchor. LLM chịu trách nhiệm sinh code heuristic mới từ các anchor đó.

## 7. Cập nhật population và memory

Heuristic mới sau khi sinh ra không được tin ngay theo mô tả của LLM. Nó phải được chạy qua evaluator thật để lấy:

```text
HV, runtime, novelty
```

Sau đó score được lưu lại:

```text
score = (-HV, runtime, -novelty)
```

Nếu heuristic mới tốt hoặc bổ sung được hướng tìm kiếm có ích, nó được đưa vào population/archive. Sau đó hệ thống cập nhật lại toàn bộ `M[lambda]` bằng cách chọn lại các set đại diện cho từng preference.

## 8. Tóm tắt luồng chạy

```text
1. Sinh một số heuristic ban đầu bằng LLM.
2. Đánh giá thật từng heuristic.
3. Cập nhật population/archive.
4. Với mỗi preference lambda, chọn set heuristic đại diện M[lambda].
5. Chọn lambda_t đang yếu hoặc lâu chưa cải thiện.
6. Lấy M[lambda_t] làm anchor.
7. Gọi LLM sinh heuristic mới.
8. Đánh giá heuristic mới bằng evaluator.
9. Cập nhật population và memory.
10. Lặp lại đến khi hết budget.
```

## 9. Khác biệt với MPaGE

MPaGE chủ yếu duy trì một Pareto population chung và chọn parent từ population đó. LLM-PSL bổ sung thêm bộ nhớ theo preference:

```text
lambda -> set heuristic đại diện
```

Nhờ đó, LLM-PSL biết vùng đánh đổi nào đang yếu, vùng nào lâu chưa được cải thiện, và nên lấy heuristic nào làm anchor để sinh chương trình mới. Đây là điểm giúp phương pháp chủ động bao phủ nhiều vùng Pareto hơn thay vì chỉ khai thác quanh các heuristic đang tốt trong population chung.
