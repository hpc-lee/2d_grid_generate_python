import pstats

stats = pstats.Stats("./output/profiles/profile_rank_0.log")

stats.sort_stats("cumulative").print_stats(10)