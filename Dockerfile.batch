FROM nvcr.io/nvidia/pytorch:25.04-py3

WORKDIR /app

COPY pyproject.toml ./
COPY batch_invariant_ops/ ./batch_invariant_ops/

RUN pip install --no-cache-dir -e .

COPY test_batch_invariance.py ./

CMD ["python", "test_batch_invariance.py"]