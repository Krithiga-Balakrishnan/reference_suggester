import json
from semantic_search_api import search_similar_papers_all_sentences
from collections import defaultdict

def load_queries(path):
    return [json.loads(line) for line in open(path)]

def load_qrels(path):
    qrels = defaultdict(dict)
    for line in open(path):
        obj = json.loads(line)
        qrels[obj['query_id']][obj['paper_id']] = obj['relevance']
    return qrels


def precision_at_k(results, qrel, k):
    topk = results[:k]
    rels = [qrel.get(r['paper_id'], 0) for r in topk]
    return sum(rels)/k


def recall_at_k(results, qrel, k):
    relevant_ids = {pid for pid, rel in qrel.items() if rel > 0}
    found = {r['paper_id'] for r in results[:k] if r['paper_id'] in relevant_ids}
    return len(found)/len(relevant_ids) if relevant_ids else 0


def reciprocal_rank(results, qrel):
    for idx, r in enumerate(results, start=1):
        if qrel.get(r['paper_id'], 0) > 0:
            return 1/idx
    return 0


def run_evaluation(query_file, qrel_file, top_k=10):
    queries = load_queries(query_file)
    qrels = load_qrels(qrel_file)
    metrics = {'P@10': [], 'R@10': [], 'RR': []}
    for q in queries:
        hits = search_similar_papers_all_sentences(q['text'], top_k=top_k)
        qrel = qrels[q['query_id']]
        metrics['P@10'].append(precision_at_k(hits, qrel, top_k))
        metrics['R@10'].append(recall_at_k(hits, qrel, top_k))
        metrics['RR'].append(reciprocal_rank(hits, qrel))
    print(f"Mean P@10: {sum(metrics['P@10'])/len(metrics['P@10']):.4f}")
    print(f"Mean R@10: {sum(metrics['R@10'])/len(metrics['R@10']):.4f}")
    print(f"MRR: {sum(metrics['RR'])/len(metrics['RR']):.4f}")