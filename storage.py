# -*- coding: utf-8 -*-
"""
storage.py
==========
تخزين بسيط فى ملفات JSON للمستخدمين والبلاغات - كفاية لعرض توضيحي (demo).
كلمات السر مشفّرة (PBKDF2) مش نص عادى.
"""

import hashlib
import json
import os
import secrets
from datetime import datetime, timezone


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()


class JSONStore:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            self._write({})

    def _read(self):
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class UserStore(JSONStore):
    def __init__(self, path):
        super().__init__(path)
        self._seed_demo_decision_account()

    def _seed_demo_decision_account(self):
        data = self._read()
        seed_accounts = [
            ("admin@roadwise.eg", "مسؤول التخطيط", "Admin@123"),
            ("rofaida.alqassas@gmail.com", "rofaida amr", "DIGI@2026"),
            # حساب ديمو تاني لعرض لوحة متخذ القرار - نفس فكرة الحساب اللي فوق.
            ("test1@gmail.com", "test1", "DIGI@2026"),
        ]
        changed = False
        for email, name, password in seed_accounts:
            if email not in data:
                salt = secrets.token_hex(16)
                data[email] = {
                    "name": name,
                    "email": email,
                    "role": "decision",
                    "salt": salt,
                    "password_hash": _hash_password(password, salt),
                    "national_id": None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                changed = True
        if changed:
            self._write(data)

    def get(self, email: str):
        return self._read().get(email)

    def create(self, name, email, password, role, national_id=None):
        data = self._read()
        salt = secrets.token_hex(16)
        user = {
            "name": name,
            "email": email,
            "role": role,
            "salt": salt,
            "password_hash": _hash_password(password, salt),
            "national_id": national_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        data[email] = user
        self._write(data)
        return user

    def verify_password(self, user: dict, password: str) -> bool:
        return _hash_password(password, user["salt"]) == user["password_hash"]


class IncidentStore(JSONStore):
    def create(self, governorate, road_name, description, image_path, reported_by,
               image_verification=None, report_verification=None):
        data = self._read()
        incident_id = f"INC-{len(data) + 1:06d}"
        record = {
            "id": incident_id,
            "governorate": governorate,
            "road_name": road_name,
            "description": description,
            "image_path": image_path,
            "reported_by": reported_by,
            # نتيجة فحص الصورة (image_verification.py) - قد تكون None لو
            # مفيش صورة أصلاً في البلاغ. مش بتغيّر حالة "status" تلقائيًا،
            # دي معلومة إضافية لعرضها في لوحة متخذ القرار.
            "image_verification": image_verification,
            # الحكم المدمج (نص + صورة) من Report Verification AI - راجع
            # main.py:_combine_verification. برضه مش بيغيّر status تلقائيًا؛
            # القرار النهائي بيفضل لمتخذ القرار عن طريق /review-incident.
            "report_verification": report_verification,
            "classification": (report_verification or {}).get("overall_classification", "needs_review"),
            # status بالإنجليزي (pending/verified/false/rejected) عشان يتوافق
            # مباشرة مع فلتر الواجهة (complaintFilter) وأزرار المراجعة.
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        data[incident_id] = record
        self._write(data)
        return record

    def list_all(self):
        return list(self._read().values())

    def get(self, incident_id: str):
        return self._read().get(incident_id)

    def update(self, incident_id: str, **fields):
        data = self._read()
        record = data.get(incident_id)
        if record is None:
            return None
        record.update(fields)
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        data[incident_id] = record
        self._write(data)
        return record

    def list_by_user(self, reported_by: str):
        return [r for r in self._read().values() if r.get("reported_by") == reported_by]