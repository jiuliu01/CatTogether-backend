"""Stage 7 verification: 12 items from word/5.Memory2.2实施方案.md §10.

Run: python backend/scripts/v22_verify.py
Requires: Qdrant at 127.0.0.1:6333 + Catenv (qdrant-client, fastembed, sentence-transformers).
"""
import sys, os, asyncio, uuid, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("CT_MEMORY_V22_ENABLED", "true")

import importlib, config
importlib.reload(config)

results = {}
def check(name, ok, detail=""):
    results[name] = (ok, detail)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

import httpx

# ---- 1. Qdrant up + collection ready ----
r = httpx.get("http://127.0.0.1:6333/collections", timeout=5)
cols = [c["name"] for c in r.json()["result"]["collections"]]
check("1.qdrant_up", "memories" in cols, f"collections={cols}")

from bootstrap import register_builtins
register_builtins()
from core.registry import registry
check("1b.memory_agent_registered", registry.get("memory") is not None, "")

from memory.v22 import get_memory_store, get_memory_extractor, reset_for_tests
reset_for_tests()
store = get_memory_store()
check("1c.store_ready", store is not None, "")

ci = httpx.get("http://127.0.0.1:6333/collections/memories").json()["result"]
sparse_keys = list(ci["config"]["params"].get("sparse_vectors", {}).keys())
check("1d.sparse_slot", "bm25" in sparse_keys, f"sparse={sparse_keys}")

# ---- 2. memory agent extraction (mock agent emits JSONL) ----
from memory.v22.extractor import MemoryExtractor, ExtractionContext
from memory.v22.models import Memory, MemorySearchResult


class MockMemAgent:
    _spec = type("S", (), {"timeout": 30, "system_prompt": ""})()

    def __init__(self, link_id=None):
        self.link_id = link_id

    async def invoke(self, ctx):
        # Build JSON with json.dumps to avoid f-string brace escaping headaches.
        mem0 = {
            "id": "0",
            "text": "a new distinct memory about vector indexing",
            "attributed_to": "assistant",
            "linked_memory_ids": [self.link_id] if self.link_id else [],
        }
        mem1 = {
            "id": "1",
            "text": "user prefers replies in chinese",
            "attributed_to": "user",
            "linked_memory_ids": [],
        }
        out = json.dumps({"memory": [mem0, mem1]}, ensure_ascii=False)
        yield ("final_text", {"text": out})


# ---- 3+4+5+9: write / dual-path / domain filter / dedup ----
from qdrant_client import QdrantClient
from fastembed import SparseTextEmbedding
from memory.embedder import get_embedder

client = QdrantClient(url="http://127.0.0.1:6333")
dense = get_embedder()
sparse = SparseTextEmbedding(model_name="Qdrant/bm25")
verify_col = "memories_verify_" + uuid.uuid4().hex[:6]
vstore = type(store)(client, dense, sparse, collection_name=verify_col, vector_size=512)
vstore.ensure_collection()

m1 = Memory.create(text="project uses BGE dense vectors for semantic search",
                   domain="project", attributed_to="assistant", run_id="rv1")
m2 = Memory.create(text="user prefers replies in chinese", domain="user", attributed_to="user")
vstore.upsert(m1)
vstore.upsert(m2)
time.sleep(1.5)

cnt = client.count(collection_name=verify_col).count
check("3.write", cnt == 2, f"points={cnt}")

hits = vstore.search("semantic vector search", domains=["project"], top_k=5)
ids_hit = [h.memory.id for h in hits]
signals_all = set()
for h in hits:
    signals_all.update(h.signals)
# dual-path passes if m1 is found AND both signals fired somewhere.
check("4.dual_path", m1.id in ids_hit and "vector" in signals_all and "bm25" in signals_all,
      f"hits={len(hits)} signals={signals_all}")

hits_user = vstore.search("vector", domains=["user"], top_k=5)
proj_leaked = any(h.memory.id == m1.id for h in hits_user)
check("5.domain_filter", not proj_leaked, f"project_in_user={proj_leaked}")

vstore.upsert(m1)
cnt2 = client.count(collection_name=verify_col).count
dup = vstore.exists_by_hash("project", m1.text)
check("9.dedup", dup and cnt2 == 2, f"exists_by_hash={dup} points={cnt2}")

# ---- 6. two render forms ----
from memory.v22.render import MemoryRenderer
sr = [MemorySearchResult(memory=m1, score=0.9, signals=["vector", "bm25"])]
extraction_text = MemoryRenderer.for_extraction(sr)
prompt_text = MemoryRenderer.for_prompt(sr)
tool_dict = MemoryRenderer.for_tool(sr)
check("6.render_extraction", "uuid=" in extraction_text and "linked=" in extraction_text, "has uuid+linked")
check("6.render_prompt", "uuid=" not in prompt_text and m1.text in prompt_text, "text only no uuid")
check("6.render_tool", tool_dict[0]["text"] == m1.text and "id" in tool_dict[0], "dict text+id")

# ---- 7+8: auto-trigger (mock run) + linked_memory_ids ----
# The mock agent emits a NEW memory (distinct text) that links to m1.
# The extractor's related-search must surface m1 so its uuid is in offered_uuids.
ext = MemoryExtractor(vstore, MockMemAgent(link_id=m1.id), history=None,
                      recent_turns=5, related_top_k=3)
ctx = ExtractionContext(
    event_type="agent_invocation",
    agent_id="coder", agent_name="coder",
    user_text="how does vector search work",
    final_text="project uses BGE dense vectors for semantic search",
    mutation_paths=["backend/memory/v22/memory_store.py"],
    succeeded=True,
    run_id="run-mock-1",
    channel_id="ch-mock", thread_id="th-mock",
    project_id="proj-mock",
    user_id="feishu:t1:u1",
)
r = asyncio.run(ext.extract(ctx))
check("7.auto_trigger", len(r.written) >= 1,
      f"written={len(r.written)} dup={r.skipped_dup} invalid={r.skipped_invalid} failed={r.failed}")

linked_found = False
for w in r.written:
    if m1.id in w.linked_memory_ids:
        linked_found = True
check("8.linked_ids", linked_found, f"m1.id={m1.id[:12]} linked={linked_found}")

if r.written:
    pts = client.retrieve(collection_name=verify_col, ids=[w.id for w in r.written], with_payload=True)
    for p in pts:
        if p.payload.get("linked_memory_ids"):
            check("8b.linked_in_qdrant", m1.id in p.payload["linked_memory_ids"],
                  f"payload={p.payload['linked_memory_ids']}")
            break

# ---- 10. switch off ----
reset_for_tests()
os.environ["CT_MEMORY_V22_ENABLED"] = "false"
importlib.reload(config)
from memory.v22 import get_memory_store as gms
s_off = gms()
check("10.switch_off", s_off is None, f"store={s_off}")

# ---- 11. failure non-fatal ----
os.environ["CT_MEMORY_V22_ENABLED"] = "true"
importlib.reload(config)


class DeadStore:
    def ensure_collection(self): pass
    def upsert(self, m): raise RuntimeError("qdrant dead")
    def exists_by_hash(self, d, t): raise RuntimeError("qdrant dead")
    def search(self, q, **kw): raise RuntimeError("qdrant dead")


ext_dead = MemoryExtractor(DeadStore(), MockMemAgent(), history=None)
r_dead = asyncio.run(ext_dead.extract(ctx))
check("11.failure_nonfatal", r_dead.failed is not None or len(r_dead.written) == 0,
      f"failed={r_dead.failed} written={len(r_dead.written)}")

# ---- 12. internal agent not usable ----
from agents.memory_agent import is_internal
check("12.internal_isolated", is_internal("memory") and not is_internal("claude-code"),
      f"memory_internal={is_internal('memory')}")

# ---- cleanup ----
client.delete_collection(verify_col)

print("\n" + "=" * 60)
passed = sum(1 for ok, _ in results.values() if ok)
total = len(results)
print(f"STAGE 7: {passed}/{total} PASSED")
for name, (ok, detail) in results.items():
    if not ok:
        print(f"  FAILED: {name} — {detail}")
