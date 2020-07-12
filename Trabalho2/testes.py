start = datetime.now()
   for i in range(0, 10):
        comparador = Trabalho(
            dataFilePath="Datasets/%s/%s.csv", datasetIndex=i)
        comparador.default_rotine()
    print(datetime.now()-start)
