import numpy as np

'''
Temporary solution for dummy data
'''

def create_depth_table():

    # This is recorded and fixed data where no self-occlusion takes place. 
    original_recorded_depth = [[2.8420002460479736, 2.9170000553131104, 3.011000156402588, 2.8940000534057617, 2.879000186920166, 2.9010000228881836, 2.813000202178955, 2.7710001468658447, 2.744000196456909, 2.821000099182129, 2.879000186920166, 2.947000026702881, 2.806000232696533, 2.9170000553131104, 2.9090001583099365, 2.806000232696533, 2.806000232696533, 2.9010000228881836, 2.871000051498413, 2.955000162124634, 2.886000156402588, 2.9, 2.9, 2.9090001583099365, 2.9010000228881836],
            [2.765000104904175, 2.8490002155303955, 2.886000156402588, 2.751000165939331, 2.4070000648498535, 2.8420002460479736, 2.758000135421753, 2.4760000705718994, 2.8350000381469727, 2.9010000228881836, 2.932000160217285, 2.9790000915527344, 2.8490002155303955, 2.871000051498413, 2.9, 2.744000196456909, 2.7310001850128174, 2.821000099182129, 2.7920000553131104, 3, 3, 3, 3, 3.052000045776367, 2.940000057220459],
            [2.8570001125335693, 2.871000051498413, 2.9240000247955322, 2.7170002460479736, 2.515000104904175, 2.886000156402588, 2.806000232696533, 2.5320000648498535, 2.813000202178955, 2.879000186920166, 2.932000160217285, 2.947000026702881, 2.828000068664551, 2.932000160217285, 2.9, 2.886000156402588, 2.828000068664551, 3.003000259399414, 2.9170000553131104, 2.9, 2.9240000247955322, 2.9, 2.9, 2.9, 2.9790000915527344],
            [2.828000068664551, 2.8490002155303955, 2.9790000915527344, 2.932000160217285, 2.871000051498413, 2.886000156402588, 2.8490002155303955, 2.8420002460479736, 2.7170002460479736, 2.765000104904175, 2.806000232696533, 2.955000162124634, 2.7310001850128174, 2.7920000553131104, 3, 2.8490002155303955, 2.7920000553131104, 2.9010000228881836, 2.8570001125335693, 3, 3, 3, 3, 3, 2.9870002269744873],
            [2.7920000553131104, 2.886000156402588, 2.940000057220459, 2.9170000553131104, 2.9, 2.6, 2.628000020980835, 2.813000202178955, 2.7170002460479736, 2.7990000247955322, 2.8490002155303955, 2.9240000247955322, 2.7240002155303955, 2.871000051498413, 2.9, 2.7990000247955322, 2.806000232696533, 2.932000160217285, 2.879000186920166, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9010000228881836],
            [2.6650002002716064, 2.765000104904175, 2.747000026702881, 2.9710001945495605, 2.765000104904175, 2.758000135421753, 2.864000082015991, 2.9240000247955322, 2.6650002002716064, 2.698000192642212, 2.7310001850128174, 2.879000186920166, 2.68500018119812, 2.821000099182129, 2.93, 2.68500018119812, 2.6590001583099365, 2.765000104904175, 2.7170002460479736, 2.9010000228881836, 2.95, 2.9, 2.932, 2.9, 2.955000162124634],
            [2.6090002059936523, 2.704000234603882, 2.7110002040863037, 2.8490002155303955, 3, 2.68500018119812, 2.821000099182129, 2.9950002002716064, 2.6460001468658447, 2.6720001697540283, 2.7710001468658447, 2.886000156402588, 2.6650002002716064, 2.758000135421753, 2.8842, 2.6530001163482666, 2.621000051498413, 2.698000192642212, 2.6650002002716064, 2.8570001125335693, 2.9, 2.828000068664551, 2.9, 2.9, 2.9170000553131104],
            [2.864000082015991, 2.8490002155303955, 2.871000051498413, 2.573000192642212, 2.3, 2.806000232696533, 2.5380001068115234, 2.311000108718872, 2.698000192642212, 2.7110002040863037, 2.806000232696533, 2.947000026702881, 2.7170002460479736, 2.7920000553131104, 2.9, 2.879000186920166, 2.864000082015991, 2.932000160217285, 2.9710001945495605, 2.864000082015991, 2.9, 2.9, 2.9, 2.9, 2.947000026702881],
            [2.6530001163482666, 2.765000104904175, 2.886000156402588, 2.9010000228881836, 2.93467, 2.7850000858306885, 2.879000186920166, 2.9240000247955322, 2.68500018119812, 2.7170002460479736, 2.828000068664551, 2.7920000553131104, 2.6780002117156982, 2.7710001468658447, 2.6, 2.6530001163482666, 2.6460001468658447, 2.7310001850128174, 2.7240002155303955, 2.9, 2.9, 2.9, 2.9, 2.9, 2.9010000228881836],
            [2.8570001125335693, 2.871000051498413, 2.879000186920166, 2.7170002460479736, 2.7110002040863037, 2.947000026702881, 2.940000057220459, 2.940000057220459, 2.7240002155303955, 2.813000202178955, 2.8420002460479736, 2.8940000534057617, 2.7780001163482666, 2.8570001125335693, 2.9, 2.879000186920166, 2.8570001125335693, 2.955000162124634, 2.9010000228881836, 2.8570001125335693, 2.8570001125335693, 2.9, 2.9, 2.9, 2.9010000228881836],
            [2.8490002155303955, 2.84900021553039554, 2.886000156402588, 2.744000196456909, 2.3460001945495605, 2.8570001125335693, 2.498000144958496, 2.3310000896453857, 2.6590001583099365, 2.7240002155303955, 2.744000196456909, 2.9170000553131104, 2.698000192642212, 2.813000202178955, 2.85, 2.886000156402588, 2.879000186920166, 2.9710001945495605, 2.947000026702881, 2.9240000247955322, 2.864000082015991, 2.947000026702881, 2.846, 2.876, 2.8490002155303955],
            [2.810000123977661, 2.7990000247955322, 2.498000144958496, 2.7310001850128174, 2.515000104904175, 2.5530001163482666, 2.5260000228881836, 2.3460001945495605, 2.704000234603882, 2.828000068664551, 2.821000099182129, 2.9090001583099365, 2.765000104904175, 2.821000099182129, 2.88, 2.4650001525878906, 2.567000150680542, 2.751000165939331, 2.6090002059936523, 2.940000057220459, 2.8420002460479736, 2.8940000534057617, 3.0, 3.078000068664551, 2.871000051498413],
            [2.744000196456909, 2.8420002460479736, 2.9170000553131104, 2.7170002460479736, 2.515000104904175, 2.813000202178955, 2.6780002117156982, 2.4650001525878906, 2.8406000137329101, 2.515000104904175, 2.7370002269744873, 2.8940000534057617, 2.4170000553131104, 2.751000165939331, 2.9, 2.758000135421753, 2.7240002155303955, 2.8570001125335693, 2.7920000553131104, 3.0860002040863037, 2.8940000534057617, 2.9090001583099365, 2.9, 2.9, 2.9],
            [2.90, 2.9010000228881836, 2.9240000247955322, 2.806000232696533, 2.5440001487731934, 2.8940000534057617, 2.640000104904175, 2.6460001468658447, 2.751000165939331, 2.7920000553131104, 2.9240000247955322, 2.947000026702881, 2.7710001468658447, 2.8420002460479736, 2.879000186920166, 2.7910000801086426, 2.879000234603882, 2.844000196456909, 2.85910000801086426, 2.864000082015991, 2.955000162124634, 2.932000160217285, 2.9, 2.9, 2.886000156402588],
            [2.8710000514984133, 2.871000051498413, 2.864000082015991, 2.744000196456909, 2.5490000247955322, 2.864000082015991, 2.6030001640319824, 2.5210001468658447, 2.7240002155303955, 2.758000135421753, 2.871000051498413, 2.9010000228881836, 2.751000165939331, 2.821000099182129, 2.9, 2.4700000286102295, 2.75, 2.75, 2.75, 2.75, 2.879000186920166, 2.9010000228881836, 2.9, 2.9, 2.9240000247955322],
            [2.8070001220703125, 2.806000232696533, 2.8060002517700195, 2.947000026702881, 2.821000099182129, 2.813000202178955, 2.5850000381469727, 2.5850000381469727, 2.704000234603882, 2.7240002155303955, 2.879000186920166, 2.8940000534057617, 2.7310001850128174, 2.7920000553131104, 2.9010000228881836, 2.8090001583099365, 2.8070001220703125, 2.80500018119812, 2.8050000381469727, 2.864000082015991, 2.864000082015991, 2.955000162124634, 2.9, 2.9, 2.9090001583099365],
            [2.801000099182129, 2.7920000553131104, 2.7540002155303955, 2.7110002040863037, 2.438000202178955, 2.831000051498413, 2.871000051498413, 2.879000186920166, 2.7110002040863037, 2.765000104904175, 2.871000051498413, 2.886000156402588, 2.7110002040863037, 2.7370002269744873, 2.806000232696533, 2.8050000190734863, 2.8060001373291016, 2.804000186920166, 2.804000234603882, 2.8420002460479736, 2.88, 2.871000051498413, 2.91, 2.92, 2.9240000247955322],
            [2.4700000286102295, 2.5380001068115234, 2.6090002059936523, 2.632000160217285, 2.8420002460479736, 2.579000234603882, 2.5440001487731934, 2.567000150680542, 2.4810001850128174, 2.509000062942505, 2.6650002002716064, 2.7370002269744873, 2.515000104904175, 2.68500018119812, 2.751000165939331, 2.492000102996826, 2.4760000705718994, 2.561000108718872, 2.5260000228881836, 2.7780001163482666, 2.7710001468658447, 2.7710001468658447, 2.7370002269744873, 2.7110002040863037, 2.7310001850128174]]

    # Convert it to numpy array
    depth_info = np.array([np.array(xi) for xi in original_recorded_depth])

    # Just some scalaras to make dummy data better for different distances from camera
    scalars = [0.65]

    depth_matrix = []

    for sca in scalars:
        for di in depth_info:
            tmp = di*sca
            depth_matrix.append(tmp)

    return depth_matrix 

