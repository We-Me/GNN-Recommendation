import math

class Metric:
    eps = 1e-10  # 防止除以0的小常数

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            if user not in res:
                continue
            items = set(origin[user].keys())
            predicted = {item[0] for item in res[user]}
            hit_count[user] = len(items & predicted)
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        total_num = sum(len(items) for items in origin.values())
        hit_num = sum(hits.get(user, 0) for user in origin)
        return round(hit_num / (total_num + Metric.eps), 5)

    @staticmethod
    def precision(hits, n):
        total_hits = sum(hits.values())
        return round(total_hits / (len(hits) * n + Metric.eps), 5)

    @staticmethod
    def recall(hits, origin):
        recall_list = []
        for user in hits:
            if user in origin:
                denominator = len(origin[user]) + Metric.eps
                recall_list.append(hits[user] / denominator)
        return round(sum(recall_list) / (len(recall_list) + Metric.eps), 5)

    @staticmethod
    def ndcg(origin, res, n):
        sum_ndcg = 0
        valid_users = 0
        for user in res:
            if user not in origin:
                continue
            dcg = 0
            idcg = 0
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    dcg += 1.0 / math.log2(n + 2)
            for n in range(min(len(origin[user]), n)):
                idcg += 1.0 / math.log2(n + 2)
            if idcg > 0:
                sum_ndcg += dcg / (idcg + Metric.eps)
                valid_users += 1
        return round(sum_ndcg / (valid_users + Metric.eps), 5)


def ranking_evaluation(origin, res, n_list):
    if len(origin) != len(res):
        raise ValueError('The Lengths of test set and predicted set do not match!')

    measures = {}
    for n in n_list:
        predicted = {user: res[user][:n] for user in res}
        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_ratio(origin, hits)
        precision = Metric.precision(hits, n)
        recall = Metric.recall(hits, origin)
        ndcg = Metric.ndcg(origin, predicted, n)
        measures[n] = {
            'Hit Ratio': hr,
            'Precision': precision,
            'Recall': recall,
            'NDCG': ndcg
        }
    return measures
