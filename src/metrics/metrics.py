import math

class Metric:
    eps = 1e-10

    @staticmethod
    def hits(gt: dict, rec: dict, k: int) -> dict:
        hit_count = {}
        for u, gt_items in gt.items():
            gt_set = set(gt_items)  # 这里假设 gt[u] 是 set/list；若是 dict，请改成 set(gt_items.keys())
            pred = rec.get(u, [])[:k]
            hit_count[u] = len(gt_set.intersection(pred))
        return hit_count

    @staticmethod
    def precision(hits: dict, num_users: int, k: int) -> float:
        total_hits = sum(hits.values())
        return total_hits / (num_users * k + Metric.eps)

    @staticmethod
    def recall(gt: dict, hits: dict) -> float:
        s = 0.0
        c = 0
        for u, gt_items in gt.items():
            denom = len(gt_items)
            if denom == 0:
                continue
            s += hits.get(u, 0) / (denom + Metric.eps)
            c += 1
        return s / (c + Metric.eps)

    @staticmethod
    def ndcg(gt: dict, rec: dict, k: int) -> float:
        sum_ndcg = 0.0
        valid = 0
        for u, gt_items in gt.items():
            gt_set = set(gt_items)
            if len(gt_set) == 0:
                continue
            pred = rec.get(u, [])[:k]

            dcg = 0.0
            for rank, item in enumerate(pred):
                if item in gt_set:
                    dcg += 1.0 / math.log2(rank + 2)

            ideal_len = min(len(gt_set), k)
            if ideal_len == 0:
                continue
            idcg = sum(1.0 / math.log2(r + 2) for r in range(ideal_len))

            sum_ndcg += dcg / (idcg + Metric.eps)
            valid += 1

        return sum_ndcg / (valid + Metric.eps)
    