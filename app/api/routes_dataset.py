from fastapi import APIRouter

from app.database import repository

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("")
def list_datasets() -> list[dict]:
    return repository.list_datasets()


@router.get("/{dataset_id}")
def get_dataset(dataset_id: int) -> dict:
    return repository.get_dataset(dataset_id)


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int) -> dict:
    repository.delete_dataset(dataset_id)
    return {"deleted": dataset_id}
