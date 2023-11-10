from scipy import stats


def get_percentile(predictions, prediction):
    rank = stats.percentileofscore(predictions, prediction)
    return round(100 - round(rank, 2), 2)


def prof(func):
    def new_func(*args, **kwargs):
        import cProfile
        import io
        import pstats
        import gc
        pr = cProfile.Profile()
        pr.enable()
        print('Profiling: \n')
        result = func(*args, **kwargs)

        pr.disable()
        pr.print_stats()
        stream = io.StringIO()
        ps = pstats.Stats(pr, stream=stream)
        ps.sort_stats('tottime').print_stats(0.1)
        stream.seek(0)
        print(stream.read())

        return result

    return new_func
