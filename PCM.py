import matplotlib.pyplot as plt
import numpy as np



Time = [
    [5.225806,
    1.325516,
    0.908514,
    0.752444,
    0.775127,
    0.859155,
    1.2263,
    2.18307,
    3.39432
    ],
        [42.231003,
    11.361906,
    7.351413,
    6.380495,
    5.764558,
    5.602502,
    7.39706,
    7.114561,
    9.44937
    ],
        [145.832089,
    41.218195,
    28.70124,
    24.060825,
    22.010319,
    20.095158,
    19.729173,
    24.639041,
    31.565644,
        ],
        ]

Sp = [
        [1,
    3.942469197,
    5.752036843,
    6.945109536,
    6.741870687,
    6.08249501,
    4.261441735,
    2.393787648,
    1.539573759
        ],
        [1,
    3.716894243,
    5.744610322,
    6.618765942,
    7.325974168,
    7.537882717,
    5.709160531,
    5.935855072,
    4.469187152
        ],
        [
    1,
    3.538051314,
    5.081037927,
    6.060976255,
    6.6256236,
    7.257076008,
    7.39169802,
    5.918740466,
    4.619962419
        ],
    ]

Ep = [
        [1,
    0.985617299,
    0.639115205,
    0.434069346,
    0.269674827,
    0.168958195,
    0.066585027,
    0.023937876,
    0.010691484,
    ],
        [1,
    0.929223561,
    0.638290036,
    0.413672871,
    0.293038967,
    0.209385631,
    0.089205633,
    0.059358551,
    0.031036022,
    ],
        [1,
    0.884512829,
    0.56455977,
    0.378811016,
    0.265024944,
    0.201585445,
    0.115495282,
    0.059187405,
    0.032083072,
    ],
      ]



Metrics = ["Total Time", "S(n,p)", "E(n,p)"]


P_num = ["1", "4", "9", "16", "25", "36", "64", "100", "144"]


for i, mt in enumerate(Metrics):
    plt.figure(figsize=(8, 8))
    plt.title(mt, fontsize=20)
    if i == 0:
        plt.ylabel("Time(/second)", fontsize=15)
        M = Time
    elif i == 1:
        plt.ylabel("Speed-up Ratio", fontsize=15)
        M = Sp
    else:
        plt.ylabel("Acceleration Efficiency ratio", fontsize=15)
        M = Ep
    plt.yticks(fontsize=15)
    plt.xticks(np.arange(1, 10), P_num, rotation=45, fontsize=15)
    for id, mv in enumerate(["n=1200", "n=2400", "n=3600"]):
        plt.plot(np.arange(1,10),  M[id], label="{}".format(mv), linewidth=3, marker="o")
    plt.legend(loc=0, fontsize=15)
    plt.savefig("./{}.png".format(mt), dpi=500)
    plt.show()