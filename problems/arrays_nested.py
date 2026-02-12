import random
from .base import TestCase
from typing import List, Optional

def gen_arrays_nested(rng: random.Random, kind: Optional[str] = None) -> dict:
    # Nested loop practice
    valid_kinds = [
        "pair_sum_count",
        "duplicate_values",
        "max_pair_product",
        "count_increasing_triplets",
        "sum_unique_values",
        "zero_sum_subarrays",
        "pairs_with_diff_k"
    ]

    if kind is None:
        kind = rng.choice(valid_kinds)
    elif kind not in valid_kinds:
        raise ValueError(f"Invalid kind '{kind}'. Valid kinds: {', '.join(valid_kinds)}")

    if kind == "pair_sum_count":
        n = rng.randint(6, 25)
        arr = [rng.randint(-10, 20) for _ in range(n)]
        i, j = rng.sample(range(n), 2)
        target = arr[i] + arr[j]

        ct = 0
        for a in range(n):
            for b in range(a + 1, n):
                if arr[a] + arr[b] == target:
                    ct += 1

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + f"\n{target}\n"
        sample_out = f"{ct}\n"

        tests = []
        for _ in range(7):
            # Must be at least 2 so rng.sample(range(tn), 2) never crashes
            tn = rng.randint(2, 60)
            ta = [rng.randint(-50, 50) for _ in range(tn)]
            ai, aj = rng.sample(range(tn), 2)
            tt = ta[ai] + ta[aj]

            tct = 0
            for a in range(tn):
                for b in range(a + 1, tn):
                    if ta[a] + ta[b] == tt:
                        tct += 1

            tests.append(TestCase(
                inp=f"{tn}\n" + " ".join(map(str, ta)) + f"\n{tt}\n",
                out=f"{tct}\n"
            ))

        return {
            "id": "ARR_PAIR_CT",
            "title": "Count Pair Sums",
            "family": "arrays_nested",
            "task": "Given n integers and a target, count how many pairs (i<j) satisfy a[i] + a[j] = target.",
            "input_format": "Line 1: n\nLine 2: n integers\nLine 3: target",
            "output_format": "One integer: number of valid pairs",
            "constraints": "2 ≤ n ≤ 60, values between -50 and 50",
            "note": "Use nested loops (no maps).",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "duplicate_values":
        n = rng.randint(8, 30)
        arr = [rng.randint(1, 12) for _ in range(n)]

        def distinct_duplicates(a: List[int]) -> int:
            dd = 0
            seen = set()
            for x in a:
                if x in seen:
                    continue
                c = 0
                for y in a:
                    if y == x:
                        c += 1
                if c >= 2:
                    dd += 1
                seen.add(x)
            return dd

        ans = distinct_duplicates(arr)
        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = f"{ans}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(1, 80)
            ta = [rng.randint(1, 25) for _ in range(tn)]
            tests.append(TestCase(inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n", out=f"{distinct_duplicates(ta)}\n"))

        return {
            "id": "ARR_DUP_DISTINCT",
            "title": "Count Distinct Duplicates",
            "family": "arrays_nested",
            "task": "Given n integers, count how many distinct values appear at least twice.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "One integer: count of distinct values with frequency ≥ 2",
            "constraints": "1 ≤ n ≤ 80, values between 1 and 25",
            "note": "Use loops/nested loops only (no maps).",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "max_pair_product":
        n = rng.randint(5, 30)
        arr = [rng.randint(-20, 20) for _ in range(n)]
        max_prod = float('-inf')
        for i in range(n):
            for j in range(i + 1, n):
                max_prod = max(max_prod, arr[i] * arr[j])

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = f"{max_prod}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(2, 50)
            ta = [rng.randint(-50, 50) for _ in range(tn)]
            mp = max(ta[i]*ta[j] for i in range(tn) for j in range(i+1, tn))
            tests.append(TestCase(
                inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n",
                out=f"{mp}\n"
            ))

        return {
            "id": "ARR_MAX_PAIR_PROD",
            "title": "Maximum Pair Product",
            "family": "arrays_nested",
            "task": "Find the maximum product of any pair of numbers in the array.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "One integer: maximum pair product",
            "constraints": "2 ≤ n ≤ 50, values between -50 and 50",
            "note": "Use nested loops (no maps).",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "count_increasing_triplets":
        n = rng.randint(5, 20)
        arr = [rng.randint(1, 20) for _ in range(n)]
        ct = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if arr[i] < arr[j] < arr[k]:
                        ct += 1

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = f"{ct}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(3, 25)
            ta = [rng.randint(1, 25) for _ in range(tn)]
            tct = sum(1 for i in range(tn) for j in range(i+1, tn) for k in range(j+1, tn) if ta[i]<ta[j]<ta[k])
            tests.append(TestCase(
                inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n",
                out=f"{tct}\n"
            ))

        return {
            "id": "ARR_INCRE_TRIPLE",
            "title": "Count Increasing Triplets",
            "family": "arrays_nested",
            "task": "Count all triplets (i<j<k) where arr[i] < arr[j] < arr[k].",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "One integer: number of increasing triplets",
            "constraints": "3 ≤ n ≤ 25, values between 1 and 25",
            "note": "Use nested loops only.",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "sum_unique_values":
            n = rng.randint(5, 30)
            arr = [rng.randint(1, 15) for _ in range(n)]
            sum_unique = sum(x for x in arr if arr.count(x) == 1)

            sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
            sample_out = f"{sum_unique}\n"

            tests = []
            for _ in range(7):
                tn = rng.randint(3, 40)
                ta = [rng.randint(1, 20) for _ in range(tn)]
                su = sum(x for x in ta if ta.count(x) == 1)
                tests.append(TestCase(
                    inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n",
                    out=f"{su}\n"
                ))

            return {
                "id": "ARR_SUM_UNIQUE",
                "title": "Sum of Unique Values",
                "family": "arrays_nested",
                "task": "Sum all numbers in the array that appear exactly once.",
                "input_format": "Line 1: n\nLine 2: n integers",
                "output_format": "One integer: sum of unique elements",
                "constraints": "1 ≤ n ≤ 40, values between 1 and 20",
                "note": "Use loops only (no sets/maps for counting).",
                "sample_in": sample_in,
                "sample_out": sample_out,
                "tests": [t.__dict__ for t in tests],
            }
    
    elif kind == "zero_sum_subarrays":
        n = rng.randint(5, 25)
        arr = [rng.randint(-10, 10) for _ in range(n)]
        ct = 0
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += arr[j]
                if s == 0:
                    ct += 1

        sample_in = f"{n}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = f"{ct}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(3, 30)
            ta = [rng.randint(-15, 15) for _ in range(tn)]
            tct = 0
            for i in range(tn):
                s = 0
                for j in range(i, tn):
                    s += ta[j]
                    if s == 0:
                        tct += 1
            tests.append(TestCase(
                inp=f"{tn}\n" + " ".join(map(str, ta)) + "\n",
                out=f"{tct}\n"
            ))

        return {
            "id": "ARR_ZERO_SUM",
            "title": "Count Zero-Sum Subarrays",
            "family": "arrays_nested",
            "task": "Count all contiguous subarrays whose sum equals zero.",
            "input_format": "Line 1: n\nLine 2: n integers",
            "output_format": "One integer: number of zero-sum subarrays",
            "constraints": "1 ≤ n ≤ 30, values between -15 and 15",
            "note": "Use nested loops only.",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }

    elif kind == "pairs_with_diff_k":
        n = rng.randint(5, 25)
        arr = [rng.randint(1, 20) for _ in range(n)]
        k = rng.randint(1, 10)
        ct = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(arr[i] - arr[j]) == k:
                    ct += 1

        sample_in = f"{n} {k}\n" + " ".join(map(str, arr)) + "\n"
        sample_out = f"{ct}\n"

        tests = []
        for _ in range(7):
            tn = rng.randint(2, 30)
            tk = rng.randint(1, 10)
            ta = [rng.randint(1, 25) for _ in range(tn)]
            tct = sum(1 for i in range(tn) for j in range(i+1, tn) if abs(ta[i]-ta[j])==tk)
            tests.append(TestCase(
                inp=f"{tn} {tk}\n" + " ".join(map(str, ta)) + "\n",
                out=f"{tct}\n"
            ))

        return {
            "id": "ARR_DIFF_K",
            "title": "Count Pairs with Difference K",
            "family": "arrays_nested",
            "task": "Count all pairs (i<j) where the absolute difference |arr[i]-arr[j]| equals k.",
            "input_format": "Line 1: n k\nLine 2: n integers",
            "output_format": "One integer: number of pairs with difference k",
            "constraints": "2 ≤ n ≤ 30, 1 ≤ values ≤ 25, 1 ≤ k ≤ 10",
            "note": "Use nested loops only.",
            "sample_in": sample_in,
            "sample_out": sample_out,
            "tests": [t.__dict__ for t in tests],
        }
