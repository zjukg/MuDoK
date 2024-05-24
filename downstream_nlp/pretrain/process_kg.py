import os
import json


if __name__ == "__main__":
    triples = open("data_utils/KG-entertainment2.txt", "r").readlines()
    ent_id = {}
    rel_id = {}
    for triple in triples:
        h, r, t = triple[:-1].split('\t')
        if h not in ent_id:
            ent_id[h] = len(ent_id)
        if r not in rel_id:
            rel_id[r] = len(rel_id)
        if t not in ent_id:
            ent_id[t] = len(ent_id)
    id_ent = {}
    id_rel = {}
    for ent in ent_id:
        id_ent[ent_id[ent]] = ent
    for rel in rel_id:
        id_rel[rel_id[rel]] = rel
    triple_ids = []
    for triple in triples:
        h, r, t = triple[:-1].split('\t')
        hid = ent_id[h]
        rid = rel_id[r]
        tid = ent_id[t]
        triple_ids.append((h, r, t))
    meta_data = {
        "entity2id": ent_id,
        "id2entity": id_ent,
        "relation2id": rel_id,
        "id2relation": id_rel,
        "triples": triple_ids
    }
    json.dump(meta_data, open("datasets/Amazon/metadata.json", "w"), ensure_ascii=False)

