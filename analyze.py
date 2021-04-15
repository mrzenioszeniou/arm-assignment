#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt

from pandas.core.reshape.merge import merge

plt.close("all")

TEST_ID = "TestId"
DEVICE = "Device"
BUILD = "Build"
ML_NETWORK = "MLNetwork"
OPTIMISED = "Optimised"
FREQUENCY = "CPUFrequency (MHz)"
THREADS = "Threads"
MEMORY = "PeakMemory (MB)"
TIME = "Time (ms)"
LEGEND = ["AlexNet @ 1GHz", "AlexNet @ 2GHz", "MobileNet @ 1GHz"]


## Load data
test_results = pd.DataFrame(pd.read_pickle("TestResults.pickle"))
test_results.index = test_results[TEST_ID]
del test_results[TEST_ID]

test_info = pd.read_csv("Test Info.csv")
test_info.index = test_info[TEST_ID]
del test_info[TEST_ID]

data = (
    pd.merge(test_results, test_info, on=TEST_ID)
    .reindex(
        columns=[
            DEVICE,
            BUILD,
            ML_NETWORK,
            OPTIMISED,
            FREQUENCY,
            THREADS,
            MEMORY,
            TIME,
        ]
    )
    .sort_values(
        by=[
            DEVICE,
            BUILD,
            ML_NETWORK,
            OPTIMISED,
            FREQUENCY,
            THREADS,
            MEMORY,
            TIME,
        ]
    )
)

print(data)

data.to_csv("Data.csv")

# Device 0 and 5 Threads
dev0_5 = data[(data[DEVICE] == "Device_0") & (data[THREADS] == 5)]
per_build = [
    dev0_5[(dev0_5[ML_NETWORK] == "AlexNet") & (dev0_5[FREQUENCY] == 1000)],
    dev0_5[(dev0_5[ML_NETWORK] == "AlexNet") & (dev0_5[FREQUENCY] == 2000)],
    dev0_5[(dev0_5[ML_NETWORK] == "MobileNet") & (dev0_5[FREQUENCY] == 1000)],
]


# Time(ms) per Build
plt.title("Time (ms) per Build (5 Threads)")
for series in per_build:
    series.index = series[BUILD]
    series[TIME].plot(ylabel=TIME)
plt.legend(LEGEND)


# Optimisation Impact
pre_opt = per_build[2][per_build[2][BUILD] < 6][TIME].mean()
post_opt = per_build[2][per_build[2][BUILD] >= 6][TIME].mean()
print("Optimisation Impact:", post_opt / pre_opt, "(", pre_opt, "vs", post_opt, ")")


# Frequency Impact
one_ghz = per_build[0][TIME].mean()
two_ghz = per_build[1][TIME].mean()
print("CPU Frequency Impact:", two_ghz / one_ghz, "(", one_ghz, "vs", two_ghz, ")")


# Memory(MB) per Build
plt.figure()
plt.title("Peak Memory Consumption (MB) per Build (5 Threads)")
for series in per_build:
    series.index = series[BUILD]
    series[MEMORY].plot(ylabel=MEMORY, xticks=series[BUILD])
plt.legend(LEGEND)


# Thread Count
dev0 = data[(data[DEVICE] == "Device_0") & (data[BUILD] == 10)]
per_threads = [
    dev0[(dev0[ML_NETWORK] == "AlexNet") & (dev0[FREQUENCY] == 1000)],
    dev0[(dev0[ML_NETWORK] == "AlexNet") & (dev0[FREQUENCY] == 2000)],
    dev0[(dev0[ML_NETWORK] == "MobileNet") & (dev0[FREQUENCY] == 1000)],
]

fig, axes = plt.subplots(nrows=2, ncols=1)

axes[0].set_title("Time(ms) per # of Threads")
for series in per_threads:
    series.index = series[THREADS]
    series[TIME].plot(ylabel=TIME, xticks=series[THREADS], ax=axes[0])

axes[1].set_title("Peak Memory Consumption(MB) per # of Threads")
for series in per_threads:
    series.index = series[THREADS]
    series[MEMORY].plot(ylabel=MEMORY, xticks=series[THREADS], ax=axes[1])

plt.subplots_adjust(wspace=1.0, hspace=1.0)
lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(
    lines,
    LEGEND,
    loc="upper right",
)

# plt.show()


# Device 0 vs Device 1
dev0 = data[
    (data[BUILD] == 10)
    & (data[ML_NETWORK] == "AlexNet")
    & (data[FREQUENCY] == 1000)
    & (data[DEVICE] == "Device_0")
]
dev1 = data[
    (data[BUILD] == 10)
    & (data[ML_NETWORK] == "AlexNet")
    & (data[FREQUENCY] == 1000)
    & (data[DEVICE] == "Device_1")
]
dev0.index = dev0[THREADS]
dev1.index = dev1[THREADS]

print("Time(ms) Dev1:Dev0", (dev1[TIME] / dev0[TIME]).mean())

fig, axes = plt.subplots(nrows=2, ncols=1)

axes[0].set_title("Time(ms) per # of Threads")
dev0[TIME].plot(ylabel=TIME, xticks=series[THREADS], ax=axes[0])
dev1[TIME].plot(ylabel=TIME, xticks=series[THREADS], ax=axes[0])


axes[1].set_title("Peak Memory Consumption(MB) per # of Threads")
dev0[MEMORY].plot(ylabel=MEMORY, xticks=series[THREADS], ax=axes[1])
dev1[MEMORY].plot(ylabel=MEMORY, xticks=series[THREADS], ax=axes[1])

plt.subplots_adjust(wspace=1.0, hspace=1.0)
lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(
    lines,
    ["Device 0", "Device 1"],
    loc="upper right",
)

plt.show()