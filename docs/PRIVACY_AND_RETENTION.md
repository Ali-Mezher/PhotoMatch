# PhotoMatch Privacy, Consent, and Data-Retention Policy

**Policy owner:** Golden Camera event administrator  
**System scope:** PhotoMatch local kiosk, event indexes, and evaluation data  
**Review cycle:** Before every event and at least once per academic term

PhotoMatch processes faces and face embeddings, which are sensitive biometric
data. The system is designed to run locally: event photos, selfies, embeddings,
and identity labels must not be uploaded to cloud services or committed to Git.
This document is the operating procedure for anyone collecting, accessing, or
deleting that data. It is project guidance, not a substitute for applicable law
or Golden Camera's legal advice.

## 1. Consent before collection

The event organizer and Golden Camera must agree in writing who may be
photographed, why PhotoMatch is being used, and the deletion date. Before a
person submits a selfie, the kiosk must show a short notice that states:

> PhotoMatch uses your selfie to search this event's photos on this local
> computer. Your selfie is processed for this search and is not intentionally
> saved. A face embedding is used only in memory. Event photos and their search
> index are deleted according to the event schedule shown below. Choose Cancel
> if you do not consent; staff can provide a non-biometric alternative.

Consent must be affirmative; opening the app or attending an event is not
consent. Staff must record the notice version, event ID, date, permitted use,
retention deadline, and the organizer who approved it. Consent may be withdrawn
through the contact method displayed at the kiosk. Withdrawal requests must be
authenticated and completed within seven days or sooner when required by law.

For a minor, obtain verifiable consent from a parent or legal guardian before
accepting a selfie or identity-labelled evaluation data. If the organizer
cannot confirm guardian consent, do not use face search for that person. Always
offer manual photo lookup as the non-biometric alternative.

Evaluation data requires separate, explicit consent because it associates a
selfie and event photos with an identity label. Event-search consent alone does
not authorize evaluation or research use.

## 2. Data inventory and retention

| Data | Location | Purpose | Retention limit |
|---|---|---|---|
| Uploaded selfie | Memory only during a search | Create the query embedding | Clear after the result or when the app closes; never intentionally persist |
| Query embedding | Memory only during a search | Compare the selfie with one event | Clear after the result or when the app closes |
| Raw event photos | `data/events/<event_id>/raw/` | Build the event index and deliver purchased photos | Delete 30 days after event delivery, and no later than 90 days after the event unless a new written agreement permits longer storage |
| Face index and metadata | `data/events/<event_id>/indexed/` | Search one event | Delete with, or before, the corresponding raw event photos |
| Evaluation selfies, matches, and identity labels | `data/evaluation/<event_id>/` | Measure accuracy and timing | Delete immediately after results are accepted, no later than 30 days after the evaluation run |
| Aggregate reports without names or image paths | `data/evaluation/<event_id>/report.md` or approved report storage | Document system performance | May be retained with the project record after verifying that it cannot identify a participant |
| Application logs | `logs/`, if enabled | Diagnose failures | Keep at most 14 days; never log images, embeddings, names, or full personal paths |

The event administrator must set the deletion date before importing photos and
maintain a local retention register. When consent is withdrawn or an event is
cancelled, delete that person's labelled evaluation set immediately. Because a
shared event index does not reliably support removing one person's unlabelled
faces, rebuild the index from the still-authorized source photos when targeted
removal is required.

## 3. Local-storage and access controls

- Use a dedicated, physically controlled workstation with full-disk encryption
  enabled (BitLocker, FileVault, or LUKS) and automatic screen locking.
- Store the repository and `data/` only on an encrypted local volume. Disable
  consumer cloud-sync and automatic photo-backup software for these folders.
- Give write access only to the event administrator and indexing operator.
  Kiosk users receive no filesystem or shell access and must not use an
  administrator account.
- Use named OS accounts, strong authentication, current security updates, and
  an audit trail of staff who imported, indexed, exported, or deleted an event.
- Do not copy biometric data to personal devices, email, messaging apps,
  unencrypted USB drives, public datasets, or source control.
- Keep the kiosk offline during operation. Model dependencies must be installed
  and verified before event data is loaded.
- Position the display to prevent shoulder surfing. Export only the photos a
  participant selected; do not expose another event or unrestricted folders.
- Treat backups as additional copies governed by the same consent and deletion
  date. Prefer encrypted, access-controlled backups with a documented owner.

The `.gitignore` rules are a safety net, not an access control. Before every
commit, staff must run `git status` and confirm that no photo, embedding, index,
identity folder, or evaluation report is staged.

## 4. Deletion procedure

Only an authorized event administrator may perform deletion.

1. Close PhotoMatch and stop indexing/evaluation processes so files are not in
   use. Record the event ID, reason, requestor, operator, and deletion time.
2. Delete `data/events/<event_id>/`, including both `raw/` and `indexed/`.
3. Delete `data/evaluation/<event_id>/`, including selfies, identity labels,
   copied matches, and reports that contain names or paths.
4. Delete related logs, exports, temporary files, recycle-bin/trash contents,
   and every backup copy. Do not delete unrelated events.
5. Confirm the event and evaluation directories no longer exist, search the
   approved storage locations for the event ID, and run `git status` to verify
   that biometric artifacts were never added to source control.
6. Have a second authorized staff member verify the deletion and close the
   retention-register entry.

Normal deletion may not make data physically unrecoverable on SSDs because of
wear levelling. Use encrypted storage from the start; when a device is retired,
use the device/vendor secure-erase procedure or destroy the encryption key
(crypto-erasure) according to Golden Camera's media-disposal policy.

## 5. Incident response

If data is lost, exposed, uploaded, committed, or accessed without permission:

1. Stop processing, disconnect the affected workstation from networks, and
   preserve only the minimum logs needed to understand the incident.
2. Inform the Golden Camera privacy contact and event organizer immediately.
3. Record what data and people were affected, when it happened, and who had
   access. Do not copy exposed biometric data into the incident report.
4. Revoke credentials and shared links, remove unauthorized copies, and rotate
   secrets. If data reached Git, removing the latest commit is insufficient;
   the repository owner must purge history and treat every clone as a copy.
5. Follow applicable breach-notification duties and document corrective action
   before returning the system to service.

## 6. Pre-event and post-event checklist

Before an event:

- [ ] Organizer approval, notice version, consent method, and deletion date are recorded.
- [ ] Guardian-consent and manual-alternative procedures are ready.
- [ ] Full-disk encryption, access accounts, screen lock, and offline mode are verified.
- [ ] Event data is stored only in the approved `data/events/<event_id>/` folder.

After an event:

- [ ] Exports have been delivered only to authorized recipients.
- [ ] Evaluation data is deleted within 30 days of the completed run.
- [ ] Event photos, indexes, backups, and logs are deleted by their deadlines.
- [ ] A second staff member verifies deletion and the retention register is closed.
