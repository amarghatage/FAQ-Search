import hashlib
import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request, url_for
import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
CLIENTS_DIR = BASE_DIR / "clients"
EXCLUDED_DATA_DIRS = {".git", ".venv", "venv", "chroma_db", "templates", "__pycache__"}
CLIENT_IDS = {"abc": "11234", "buffalo": "11432"}

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Use ChromaDB's native SentenceTransformer embedding function wrapper
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

faq_collection = chroma_client.get_or_create_collection(
    name="department_faqs", embedding_function=embedding_function
)
faqs_by_id = {}
clients_by_id = {}
faq_data_signature = None


def get_faq_data_signature():
    signature = []
    client_directories = sorted(CLIENTS_DIR.iterdir()) if CLIENTS_DIR.exists() else []
    for client_dir in client_directories:
        if not client_dir.is_dir():
            continue
        for file_path in sorted(client_dir.rglob("*.json")):
            try:
                file_stat = file_path.stat()
            except OSError:
                continue
            signature.append(
                (file_path.as_posix(), file_stat.st_mtime_ns, file_stat.st_size)
            )
    return tuple(signature)


def load_faqs():
    faqs = {}
    clients = {}
    client_names_by_id = {}
    client_directories = sorted(CLIENTS_DIR.iterdir()) if CLIENTS_DIR.exists() else []
    for client_dir in client_directories:
        if not client_dir.is_dir():
            continue

        client_id = CLIENT_IDS.get(
            client_dir.name,
            "client_" + hashlib.sha256(client_dir.name.encode("utf-8")).hexdigest(),
        )
        previous_name = client_names_by_id.get(client_id)
        if previous_name and previous_name != client_dir.name:
            raise ValueError(
                f"Client ID {client_id!r} is assigned to both "
                f"{previous_name!r} and {client_dir.name!r}"
            )
        client_names_by_id[client_id] = client_dir.name
        clients[client_id] = {"name": client_dir.name, "departments": set()}

        for file_path in sorted(client_dir.rglob("*.json")):
            try:
                records = json.loads(file_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                app.logger.warning("Skipping unreadable FAQ file %s: %s", file_path, error)
                continue

            if not isinstance(records, list):
                continue

            path_parts = file_path.relative_to(client_dir).parts
            category_parts = path_parts[:-1]
            if category_parts:
                department = category_parts[0]
            else:
                department_name = file_path.stem.replace("_", " ").title()
                department = "HR" if department_name.casefold() == "hr" else department_name
            subdepartment = " / ".join(category_parts[1:])

            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    continue

                question = record.get("question")
                answer = record.get("answer")
                if not isinstance(question, str) or not isinstance(answer, str):
                    continue
                question = question.strip()
                answer = answer.strip()
                if not question or not answer:
                    continue

                source_key = f"{client_dir.name}/{file_path.relative_to(client_dir).as_posix()}:{index}:{question}"
                faq_id = hashlib.sha256(source_key.encode("utf-8")).hexdigest()
                faqs[faq_id] = {
                    "id": faq_id,
                    "question": question,
                    "answer": answer,
                    "client_id": client_id,
                    "department": department,
                    "subdepartment": subdepartment,
                }
                clients[client_id]["departments"].add(department)

    return faqs, clients


def refresh_faq_index():
    global faq_data_signature

    clients_by_id.clear()
    faqs, clients = load_faqs()
    faqs_by_id.clear()
    faqs_by_id.update(faqs)
    clients_by_id.update(
        {
            client_id: {
                "name": client["name"],
                "departments": sorted(client["departments"]),
            }
            for client_id, client in clients.items()
        }
    )

    existing_ids = faq_collection.get(include=[])["ids"]
    if existing_ids:
        faq_collection.delete(ids=existing_ids)

    if faqs_by_id:
        faq_collection.add(
            ids=list(faqs_by_id),
            documents=[faq["question"] for faq in faqs_by_id.values()],
            metadatas=[
                {
                    "client_id": faq["client_id"],
                    "department": faq["department"],
                    "subdepartment": faq["subdepartment"],
                }
                for faq in faqs_by_id.values()
            ],
        )
    faq_data_signature = get_faq_data_signature()


refresh_faq_index()


@app.before_request
def refresh_index_when_faq_files_change():
    if get_faq_data_signature() != faq_data_signature:
        refresh_faq_index()


@app.route("/")
def index():
    client_id = request.args.get("client_id", "").strip()
    client = clients_by_id.get(client_id)
    return render_template(
        "index.html",
        client_id=client_id,
        client_valid=client is not None,
        departments=client["departments"] if client else [],
    )


@app.route("/clients", methods=["GET"])
def clients():
    return jsonify(
        [
            {
                "id": client_id,
                "name": client["name"],
                "url": url_for("index", client_id=client_id),
            }
            for client_id, client in sorted(clients_by_id.items())
        ]
    )


@app.route("/suggest", methods=["GET"])
def suggest():
    query = request.args.get("q", "").strip()
    client_id = request.args.get("client_id", "").strip()
    department = request.args.get("department", "").strip()
    client = clients_by_id.get(client_id)
    if not client:
        return jsonify({"error": "A valid client_id is required"}), 400
    if department and department not in client["departments"]:
        return jsonify({"error": "Department is not available for this client"}), 400
    if not query:
        return jsonify([])

    where = {"client_id": client_id}
    matching_faqs = [
        faq
        for faq in faqs_by_id.values()
        if faq["client_id"] == client_id
        and (not department or faq["department"] == department)
    ]
    if not matching_faqs:
        return jsonify([])
    if department:
        where = {"$and": [{"client_id": client_id}, {"department": department}]}

    results = faq_collection.query(
        query_texts=[query],
        n_results=min(5, len(matching_faqs)),
        where=where,
    )
    result_ids = results.get("ids", [[]])[0]
    suggestions = []
    for faq_id in result_ids:
        faq = faqs_by_id.get(faq_id)
        if faq:
            suggestions.append(
                {
                    "id": faq_id,
                    "question": faq["question"],
                }
            )
    return jsonify(suggestions)


@app.route("/answer/<faq_id>", methods=["GET"])
def answer(faq_id):
    faq = faqs_by_id.get(faq_id)
    client_id = request.args.get("client_id", "").strip()
    department = request.args.get("department", "").strip()
    if (
        not faq
        or faq["client_id"] != client_id
        or (department and faq["department"] != department)
    ):
        return jsonify({"error": "Question not found"}), 404
    return jsonify({"question": faq["question"], "answer": faq["answer"]})


if __name__ == "__main__":
    app.run(debug=True, port=5000)