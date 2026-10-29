"""Photo storage in S3."""

import uuid
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError


class StorageError(Exception):
    """S3 rejected or didn't answer the upload. Maps to a 502."""


class PhotoStorage:
    def __init__(self, bucket: str, s3_client: Any) -> None:
        self.bucket = bucket
        self._s3 = s3_client

    def put_photo(self, user_id: int, jpeg: bytes) -> str:
        # The user id prefix makes "delete everything of user N" (account deletion,
        # PIPA requests) a single prefix delete. The random name never reveals the
        # original filename and can't collide.
        key = f"photos/{user_id}/{uuid.uuid4().hex}.jpg"
        try:
            self._s3.put_object(Bucket=self.bucket, Key=key, Body=jpeg, ContentType="image/jpeg")
        except (BotoCoreError, ClientError) as exc:
            raise StorageError(f"S3 upload failed: {exc}") from exc
        return key

    def delete(self, key: str) -> None:
        # Best effort: called while another error is already being handled, and a
        # leftover object is removed by the bucket's lifecycle rule anyway.
        try:
            self._s3.delete_object(Bucket=self.bucket, Key=key)
        except (BotoCoreError, ClientError):
            pass
