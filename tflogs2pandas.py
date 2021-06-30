#!/usr/bin/env python3

import glob
import os
from os import walk
from os.path import join, split
import pprint
import traceback

import click
import pandas as pd
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pdb


# Extraction function
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    '''
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    '''
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 500,
        "images": 4,
        "audio": 4,
        "scalars": 100,
        "histograms": 1,
        "tensors": 100,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        #tags = event_acc.Tags()["tensors"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            #event_list = event_acc.Tensors(tag)
            values = list(map(lambda x: x.value, event_list))
            #values = list(map(lambda x: tf.make_ndarray(x.tensor_proto).item(), event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def many_logs2pandas(event_paths):
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tflog2pandas(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs


@click.command()
@click.argument("logdir-or-logfile")
@click.option(
    "--write-pkl/--no-write-pkl", help="save to pickle file or not", default=False
)
@click.option(
    "--write-csv/--no-write-csv", help="save to csv file or not", default=True
)
@click.option(
    "--separate_output", help="output each event separately", default=True
)
@click.option("--out-dir", "-o", help="output directory", default=".")
def main(logdir_or_logfile: str, write_pkl: bool, write_csv: bool, out_dir: str, separate_output: bool):
    """This is a enhanced version of
    https://gist.github.com/ptschandl/ef67bbaa93ec67aba2cab0a7af47700b
    This script exctracts variables from all logs from tensorflow event
    files ("event*"),
    writes them to Pandas and finally stores them a csv-file or
    pickle-file including all (readable) runs of the logging directory.
    Example usage:
    # create csv file from all tensorflow logs in provided directory (.)
    # and write it to folder "./converted"
    tflogs2pandas.py . --write-csv --no-write-pkl --o converted
    # creaste csv file from tensorflow logfile only and write into
    # and write it to folder "./converted"
    tflogs2pandas.py tflog.hostname.12345 --write-csv --no-write-pkl --o converted
    """
    pp = pprint.PrettyPrinter(indent=4)
    if os.path.isdir(logdir_or_logfile):
        # Get all event* runs from logging_dir subdirectories
        #event_paths = glob.glob(os.path.join(logdir_or_logfile, "event*"))
        event_paths = []
        for dir_path, dir_names, file_names in walk(logdir_or_logfile):
            event_paths.extend([join(dir_path, file_name) for file_name in file_names])
    elif os.path.isfile(logdir_or_logfile):
        event_paths = [logdir_or_logfile]
    else:
        raise ValueError(
            "input argument {} has to be a file or a directory".format(
                logdir_or_logfile
            )
        )
    # Call & append
    if event_paths:
        if separate_output:
            for event_path in event_paths:
                pp.pprint("Found tensorflow logs to process:")
                pp.pprint([event_path])
                all_logs = many_logs2pandas([event_path])
                pp.pprint("Head of created dataframe")
                pp.pprint(all_logs.head())

                new_out_dir = split(event_path.replace(logdir_or_logfile, out_dir))[0]
                os.makedirs(join(new_out_dir), exist_ok=True)
                if write_csv:
                    print("saving to csv file")
                    out_file = os.path.join(new_out_dir, "events.csv")
                    print(out_file)
                    all_logs.to_csv(out_file, index=None)
                if write_pkl:
                    print("saving to pickle file")
                    out_file = os.path.join(new_out_dir, "events.pkl")
                    print(out_file)
                    all_logs.to_pickle(out_file)
        else:
            pp.pprint("Found tensorflow logs to process:")
            pp.pprint(event_paths)
            all_logs = many_logs2pandas(event_paths)
            pp.pprint("Head of created dataframe")
            pp.pprint(all_logs.head())

            os.makedirs(out_dir, exist_ok=True)
            if write_csv:
                print("saving to csv file")
                out_file = os.path.join(out_dir, "all_training_logs_in_one_file.csv")
                print(out_file)
                all_logs.to_csv(out_file, index=None)
            if write_pkl:
                print("saving to pickle file")
                out_file = os.path.join(out_dir, "all_training_logs_in_one_file.pkl")
                print(out_file)
                all_logs.to_pickle(out_file)
    else:
        print("No event paths have been found.")


if __name__ == "__main__":
    main()
