import os
import numpy as np
import pandas as pd
import mne
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from riemannian_manifold import Riemann, ProjectionCS
from autoreject import AutoReject
from tqdm.notebook import tqdm
import logging
import traceback
from joblib import Parallel, delayed


#logging
logging.basicConfig(
    filename=os.path.join('./ds005106/derivatives/preprocessing', 'preprocessing.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)


def preprocess_subject(subject_id, datapath, derivatives_path, reference, l_freq, h_freq, tmin, tmax, position=0):
    """
    Preprocessing:
    1) Raw EEG reading
    2) Reference
    3) FIR filter
    4) Fast ICA (“light” preprocessing)
    5) Noisy channel interpolation
    6) Event creation and epoching
    7) AutoReject
    8) Cov SPD + TangentSpace
    9) Saving .npz
    """

    #If you want to use Class Riemann and ProjectionCS of riemannian_manifold.py file
    '''
    n_rank = 24
    riemann_components = 1
    '''

    total_steps = 9
    out_fn = os.path.join(derivatives_path, f"sub-{subject_id}_preprocessed_tangentspace.npz")

    if os.path.exists(out_fn):
        logging.info(f"Preprocessed data for sub-{subject_id} already exists. Skipping...")
        return

    subject_path = os.path.join(datapath, f"sub-{subject_id}")
    if not os.path.exists(subject_path):
        logging.warning(f"Directory for sub-{subject_id} does not exist. Skipping...")
        return

    with tqdm(total=total_steps, desc=f"Sub-{subject_id}", position=position, leave=False) as pbar_subj:

        raw_filename = os.path.join(subject_path, 'eeg', f"sub-{subject_id}_task-fix_eeg.set")
        behavfn = os.path.join(subject_path, 'eeg', f"sub-{subject_id}_task-fix_events.tsv")

        if not (os.path.isfile(raw_filename) and os.path.isfile(behavfn)):
            pbar_subj.update(total_steps)
            logging.warning(f"Missing raw or behavioral files for sub-{subject_id}. Skipping...")
            return

        try:

            raw = mne.io.read_raw_eeglab(raw_filename, preload=True, verbose='ERROR')
            pbar_subj.update(1)

            raw.set_eeg_reference(reference, verbose='ERROR')
            pbar_subj.update(1)

            raw.filter(
                l_freq=l_freq, h_freq=h_freq,
                method='fir',
                fir_window='hamming',
                fir_design='firwin',
                n_jobs=-1,
                verbose='ERROR'
            )
            pbar_subj.update(1)

            ica = mne.preprocessing.ICA(n_components=0.99, random_state=42, max_iter='auto')
            ica.fit(raw, reject=None)
            raw = ica.apply(raw)
            pbar_subj.update(1)

            data_var = np.var(raw.get_data(), axis=1)
            threshold_low = np.percentile(data_var, 1)
            threshold_high = np.percentile(data_var, 99)
            bads = [
                ch for ch, var in zip(raw.ch_names, data_var)
                if var < threshold_low or var > threshold_high
            ]
            if len(bads) > 0:
                raw.info['bads'].extend(bads)
                raw.interpolate_bads(verbose='ERROR')
                logging.info(f"Sub-{subject_id}: Interpolated bad channels: {bads}")
            pbar_subj.update(1)

            behav = pd.read_csv(behavfn, sep='\t')
            if 'stimnum' in behav.columns:
                behav['condition_num'] = behav['stimnum']
            else:
                pbar_subj.update(total_steps - pbar_subj.n)
                logging.warning(f"Sub-{subject_id}: 'stimnum' column not found. Skipping...")
                return

            valid_events = behav[
                (behav['condition_num'].notna()) &
                (behav['condition_num'] >= 1) &
                (behav['condition_num'] <= 200)
            ]
            srate = raw.info['sfreq']
            events_mne = np.vstack([
                (valid_events['onset'] * srate).astype(int),
                np.zeros(len(valid_events), dtype=int),
                valid_events['condition_num'].astype(int)
            ]).T

            if events_mne.shape[1] != 3:
                logging.error(f"Sub-{subject_id}: Events array malformed.")
                pbar_subj.update(total_steps - pbar_subj.n)
                return
            if len(events_mne) == 0:
                pbar_subj.update(total_steps - pbar_subj.n)
                logging.warning(f"Sub-{subject_id}: No valid events. Skipping...")
                return

            event_id = {str(i): i for i in range(1, 201)}
            reject_criteria = dict(eeg=150e-6)
            epochs = mne.Epochs(
                raw, events=events_mne,
                event_id=event_id,
                tmin=tmin, tmax=tmax,
                baseline=None,
                preload=True,
                verbose='ERROR',
                reject=reject_criteria)

            pbar_subj.update(1)

            from autoreject import AutoReject
            ar = AutoReject(
                random_state=42,
                n_jobs=-1,
                verbose=False,
                n_interpolate=[1, 4],
                consensus=[0.2, 0.3],
                cv=2
            )
            epochs_clean = ar.fit_transform(epochs)

            if not isinstance(epochs_clean, mne.Epochs):
                raise TypeError(f"AutoReject returned object of type {type(epochs_clean)}")

            logging.info(f"Sub-{subject_id}: Number of epochs after AutoReject: {len(epochs_clean)}")
            if len(epochs_clean) == 0:
                pbar_subj.update(total_steps - pbar_subj.n)
                logging.warning(f"Sub-{subject_id}: All epochs rejected. Skipping...")
                return
            pbar_subj.update(1)

            data = epochs_clean.get_data()
            if data.size == 0:
                pbar_subj.update(total_steps - pbar_subj.n)
                logging.warning(f"Sub-{subject_id}: No data after cleaning. Skipping...")
                return

            if np.isnan(data).any() or np.isinf(data).any():
                pbar_subj.update(total_steps - pbar_subj.n)
                logging.error(f"Sub-{subject_id}: Data contains NaN or Inf. Skipping...")
                return
            
            #If you ant to use Class Riemann and ProjectionCS of riemannian_manifold.py file
            '''
            cov_mats = Covariances(estimator = 'oas').fit_transform(data)[:, None, :, :]
            pcs = ProjectionCS(n_rank = n_rank)
            spoc = pcs.fit(cov_mats).transform(cov_mats)
            riemann = Riemann(n = riemann_components)
            ts_features = riemann.transform(spoc)
            pbar_subj.update(1)
            '''

            cov_mats = Covariances(estimator='oas').fit_transform(data)
            ts = TangentSpace()
            ts_features = ts.fit_transform(cov_mats)
            pbar_subj.update(1)

            stimnum = epochs_clean.events[:, 2]
            onsets_sec = epochs_clean.events[:, 0] / srate
            chan_labels = epochs_clean.ch_names
            times = epochs_clean.times
            subjects_arr = np.array([subject_id]*len(stimnum))

            np.savez(
                out_fn,
                ts_features=ts_features,
                labels=stimnum,
                subjects=subjects_arr,
                chan_labels=chan_labels,
                times=times,
                onsets_sec=onsets_sec
            )
            pbar_subj.update(1)
            pbar_subj.write(f"[INFO] Finished preprocessing for sub-{subject_id}, saved to {out_fn}")
            logging.info(f"Sub-{subject_id}: Finished preprocessing, saved to {out_fn}")

        except Exception as e:
            tb = traceback.format_exc()
            pbar_subj.write(f"[ERROR] sub-{subject_id} failed: {e}\n{tb}")
            logging.error(f"Sub-{subject_id}: Failed. Error: {e}\n{tb}")
            pbar_subj.update(total_steps - pbar_subj.n)


datapath = './ds005106'
derivatives_path = os.path.join(datapath, 'derivatives', 'preprocessing')
os.makedirs(derivatives_path, exist_ok=True)

l_freq, h_freq = 0.5, 40.0  
tmin, tmax = 0.0, 0.5
reference = 'average'

subject_ids = ['{:03d}'.format(i) for i in range(1, 52)]

Parallel(n_jobs=4)(
    delayed(preprocess_subject)(
        subject_id,
        datapath,
        derivatives_path,
        reference,
        l_freq,
        h_freq,
        tmin,
        tmax,
        position=1
    ) for subject_id in subject_ids
)