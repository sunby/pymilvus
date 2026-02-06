"""
Minimal reproduction case for Milvus boolean logic bug.
BUG: (False OR False) evaluates to True under specific conditions.

Key conditions that trigger the bug:
1. A nullable VARCHAR field (c10) with NULL value
2. Multiple != comparisons ANDed together on that null field: (c10 != "X" and c10 != "Y")
3. An OR clause with a JSON field comparison
4. Combined with exists() on a non-existent JSON path
"""
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection
)

HOST = "127.0.0.1"
PORT = "19530"
COLLECTION_NAME = "milvus_logic_bug_minimal"

# Minimal test data - only fields needed to reproduce the bug
DATA_ROW = {
    'id': 1,
    'vector': [0.1] * 128,
    'c10': None,           # NULL varchar - critical for bug
    'c13': 71210,          # INT64 field used in comparison
    'meta_json': {}        # Empty JSON - triggers exists() to return False
}

# Left expression evaluates to False because:
# - c10 is NULL, so (c10 != "X" and c10 != "Y") evaluates to False (NULL comparisons)
# - Even though c13 >= 0 is True, the AND with the NULL comparison makes it False
# EXPR_LEFT = '((c10 != "X" and c10 != "Y") and (c13 >= 0 or meta_json["version"] == 0))'
EXPR_LEFT = '((c10 != "X" and c10 != "Y") and (c13 >= 0 or meta_json["version"] == 0))'

# Right expression evaluates to False because:
# - meta_json["non_exist"] doesn't exist, so exists() returns False
EXPR_RIGHT = 'exists(meta_json["non_exist"])'

# Combined: (False) OR (False) should be False, but bug causes it to return True
EXPR_COMBINED = f"({EXPR_LEFT}) OR ({EXPR_RIGHT})"


def run_proof():
    print("Milvus Boolean Logic Bug - Minimal Reproduction")
    print("=" * 50)

    connections.connect("default", host=HOST, port=PORT)

    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    # Minimal schema - only fields needed for reproduction
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="c10", dtype=DataType.VARCHAR, max_length=512, nullable=True),
        FieldSchema(name="c13", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="meta_json", dtype=DataType.JSON, nullable=True),
    ]

    schema = CollectionSchema(fields)
    col = Collection(COLLECTION_NAME, schema)

    col.insert([DATA_ROW])
    col.flush()

    index_params = {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 32, "efConstruction": 256}}
    col.create_index("vector", index_params)
    col.load()

    print(f"\nTest Data: c10=NULL, c13={DATA_ROW['c13']}, meta_json={DATA_ROW['meta_json']}")
    print(f"\nEXPR_LEFT:  {EXPR_LEFT}")
    print(f"EXPR_RIGHT: {EXPR_RIGHT}")
    print(f"COMBINED:   ({EXPR_LEFT}) OR ({EXPR_RIGHT})")

    res_left = col.query(EXPR_LEFT, output_fields=["id"])
    res_right = col.query(EXPR_RIGHT, output_fields=["id"])
    res_combined = col.query(EXPR_COMBINED, output_fields=["id"])

    is_left_true = len(res_left) > 0
    is_right_true = len(res_right) > 0
    is_combined_true = len(res_combined) > 0
    
    print(res_combined)

    print(f"\nResults:")
    print(f"  EXPR_LEFT  = {is_left_true} (expected: False)")
    print(f"  EXPR_RIGHT = {is_right_true} (expected: False)")
    print(f"  COMBINED   = {is_combined_true} (expected: False)")

    print(f"\nVerdict:")
    if not is_left_true and not is_right_true and is_combined_true:
        print("  BUG CONFIRMED: (False OR False) = True")
    else:
        print("  No bug: (False OR False) = False")


if __name__ == "__main__":
    run_proof()
