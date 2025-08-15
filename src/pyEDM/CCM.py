
# python modules
from multiprocessing import Pool

# package modules
from pandas import DataFrame, concat
from numpy import array, exp, fmax, divide, mean, nan, roll, sum, zeros, in1d, arange, isin, isnan, where
from numpy.random import default_rng

# local modules
from .Simplex import Simplex as SimplexClass
from .AuxFunc import ComputeError, IsIterable

#------------------------------------------------------------
class CCM:
    '''CCM class : Base class. Contains two Simplex instances'''

    def __init__( self,
                  dataFrame       = None,
                  columns         = "",
                  target          = "", 
                  E               = 0, 
                  Tp              = 0,
                  knn             = 0,
                  tau             = -1,
                  exclusionRadius = 0,
                  libSizes        = [],
                  sample          = 0,
                  seed            = None,
                  includeData     = False,
                  embedded        = False,
                  validLib        = [],
                  noTime          = False,
                  ignoreNan       = True,
                  verbose         = False,
                  aggMethod       = None,
                  weighted        = None,
                  num_threads     = None,
                  pred_num        = None,
                  predict_col     = False,
                  selfPredict     = False,
                  shuffleLibs = False ) :
        '''Initialize CCM.'''

        # Assign parameters from API arguments
        self.name            = 'CCM'
        self.Data            = dataFrame
        self.columns         = columns
        self.target          = target
        self.E               = E
        self.Tp              = Tp
        self.knn             = knn
        self.tau             = tau
        self.exclusionRadius = exclusionRadius
        self.libSizes        = libSizes
        self.sample          = sample
        self.seed            = seed
        self.includeData     = includeData
        self.embedded        = embedded
        self.validLib        = validLib
        self.noTime          = noTime
        self.ignoreNan       = ignoreNan
        self.verbose         = verbose
        self.pred_num = pred_num
        self.predict_col = predict_col
        self.selfPredict = selfPredict
        self.shuffleLibs = shuffleLibs

        # Set full lib & pred
        self.lib = self.pred = [ 1, self.Data.shape[0] ]

        self.CrossMapList  = None # List of CrossMap results
        self.libMeans      = None # DataFrame of CrossMap results
        self.libMeans_self = None # DataFrame of CrossMap results for self-prediction
        self.PredictStats1 = None # DataFrame of CrossMap stats
        self.PredictStats2 = None # DataFrame of CrossMap stats

        self.aggMethod = aggMethod if aggMethod is not None else mean
        self.weighted  = weighted  if weighted  is not None else True
        self.num_threads = num_threads if num_threads is not None else 2

        # Setup
        self.Validate() # CCM Method

        # Instantiate Forward and Reverse Mapping objects
        # Each __init__ calls Validate() & CreateIndices()
        # and sets up targetVec, allTime
        if self.predict_col ==True:
            predict_col = [col for col in target if 'predictant' in col][0]
        else:
            predict_col = None

        self.FwdMap = SimplexClass( dataFrame       = dataFrame,
                                    columns         = columns,
                                    target          = target, 
                                    lib             = self.lib,
                                    pred            = self.pred,
                                    E               = E, 
                                    Tp              = Tp,
                                    knn             = knn,
                                    tau             = tau,
                                    exclusionRadius = exclusionRadius,
                                    embedded        = embedded,
                                    validLib        = validLib,
                                    noTime          = noTime,
                                    ignoreNan       = ignoreNan,
                                    verbose         = verbose ,
                                    predict_col     = predict_col)
        if self.selfPredict is True:
            self.FwdMap.selfPredict = self.FwdMap.columns[0] if len(self.FwdMap.columns) == 1 else None
        else:
            self.FwdMap.selfPredict = None

        if self.predict_col ==True:
            predict_col = [col for col in columns if 'predictant' in col][0]
        else:
            predict_col = None
        self.RevMap = SimplexClass( dataFrame       = dataFrame,
                                    columns         = target,
                                    target          = columns, 
                                    lib             = self.lib,
                                    pred            = self.pred,
                                    E               = E, 
                                    Tp              = Tp,
                                    knn             = knn,
                                    tau             = tau,
                                    exclusionRadius = exclusionRadius,
                                    embedded        = embedded,
                                    validLib        = validLib,
                                    noTime          = noTime,
                                    ignoreNan       = ignoreNan,
                                    verbose         = verbose,
                                    predict_col     = predict_col)
        if self.selfPredict is True:
            self.RevMap.selfPredict = self.RevMap.columns[0] if len(self.RevMap.columns) == 1 else None
        else:
            self.RevMap.selfPredict = None


    #-------------------------------------------------------------------
    # Methods
    #-------------------------------------------------------------------

    def process_CMStats(self, CMStats):
        CMDF = []
        for libSize in CMStats.keys():
            LibSize = [libSize] * self.sample  # this libSize sample times
            libStats = CMStats[libSize]  # sample ComputeError dicts

            libStatsDF = DataFrame(libStats)
            libSizeDF = DataFrame({'LibSize': LibSize})
            libDF = concat([libSizeDF, libStatsDF], axis=1)

            CMDF.append(libDF)

        return concat(CMDF, axis=0)

    def Project( self, sequential = False) :
        '''CCM both directions with CrossMap()'''

        if self.verbose:
            print( f'{self.name}: Project()' )

        if self.num_threads == 1:
            sequential = True

        if sequential : # Sequential alternative to multiprocessing 
            FwdCM = self.CrossMap( 'FWD' )
            RevCM = self.CrossMap( 'REV' )
            self.CrossMapList = [ FwdCM, RevCM ]
        else :
            # multiprocessing Pool CrossMap both directions simultaneously
            poolArgs = [ 'FWD', 'REV' ]
            with Pool( processes = self.num_threads ) as pool :
                CrossMapList = pool.map( self.CrossMap, poolArgs )

            self.CrossMapList = CrossMapList

        FwdCM, RevCM = self.CrossMapList

        self.libMeans = \
            DataFrame( {'LibSize' : FwdCM['libRho'].keys(),
                        f"{FwdCM['columns'][0]}:{FwdCM['target'][0]}" :
                        FwdCM['libRho'].values(),
                        f"{RevCM['columns'][0]}:{RevCM['target'][0]}" :
                        RevCM['libRho'].values() } )
        self.libMeans_self=None
        if self.selfPredict is True:
            self.libMeans_self = \
                DataFrame( {'LibSize' : FwdCM['libRho_self'].keys(),
                            f"{FwdCM['columns'][0]}:{FwdCM.selfPredict}" :
                            FwdCM['libRho_self'].values(),
                            f"{RevCM['columns'][0]}:{RevCM.selfPredict}" :
                            RevCM['libRho_self'].values() } )

        if self.includeData :
            FwdCMStats = FwdCM['predictStats'] # key libSize : list of CE dicts
            RevCMStats = RevCM['predictStats']

            FwdCMStats_self = FwdCM['predictStats_self']
            RevCMStats_self = RevCM['predictStats_self']

            # Create DataFrames for each libSize
            FwdStatDF = self.process_CMStats(self, FwdCMStats)
            RevStatDF = self.process_CMStats(self, RevCMStats)
            self.PredictStats1 = FwdStatDF
            self.PredictStats2 = RevStatDF
            self.PredictStats1_self = None
            self.PredictStats2_self = None

            if self.selfPredict is True:
                FwdStatDF_self = self.process_CMStats(self, FwdCMStats_self)
                RevStatDF_self = self.process_CMStats(self, RevCMStats_self)

                self.PredictStats1_self = FwdStatDF_self
                self.PredictStats2_self = RevStatDF_self

            # FwdCMDF = []
            # for libSize in FwdCMStats.keys() :
            #     LibSize  = [libSize] * self.sample # this libSize sample times
            #     libStats = FwdCMStats[libSize]     # sample ComputeError dicts
            #
            #     libStatsDF = DataFrame( libStats )
            #     libSizeDF  = DataFrame( { 'LibSize' : LibSize } )
            #     libDF      = concat( [libSizeDF, libStatsDF], axis = 1 )
            #
            #     FwdCMDF.append( libDF )
            #
            # RevCMDF = []
            # for libSize in RevCMStats.keys() :
            #     LibSize  = [libSize] * self.sample # this libSize sample times
            #     libStats = RevCMStats[libSize]     # sample ComputeError dicts
            #
            #     libStatsDF = DataFrame( libStats )
            #     libSizeDF  = DataFrame( { 'LibSize' : LibSize } )
            #     libDF      = concat( [libSizeDF, libStatsDF], axis = 1 )
            #
            #     RevCMDF.append( libDF )
            #
            # FwdStatDF = concat( FwdCMDF, axis = 0 )
            # RevStatDF = concat( RevCMDF, axis = 0 )

            # self.PredictStats1 = FwdStatDF
            # self.PredictStats2 = RevStatDF

    #-------------------------------------------------------------------
    # 
    #-------------------------------------------------------------------
    def CrossMap( self, direction ) :
        if self.verbose:
            print( f'{self.name}: CrossMap()' )

        if direction == 'FWD' :
            S = self.FwdMap
        elif direction == 'REV' :
            S = self.RevMap
        else :
            raise RuntimeError( f'{self.name}: CrossMap() Invalid Map' )

        if S.selfPredict is not None :
            S.targetVec_self = S.dataFrame[ S.selfPredict ].to_numpy()

        if self.shuffleLibs is True:
            # Shuffle the library indices
            RNG = default_rng(self.seed)
            shuffled_inds = RNG.permutation(arange(S.targetVec))
            S.targetVec = S.targetVec[shuffled_inds]
            if S.selfPredict is not None:
                S.targetVec_self = S.targetVec_self[shuffled_inds]


        # Create random number generator : None sets random state from OS
        RNG = default_rng( self.seed )

        # Copy S.lib_i since it's replaced every iteration
        lib_i   = S.lib_i.copy() 
        N_lib_i = len( lib_i )
        knn = S.knn

        libRhoMap  = {} # Output dict libSize key : mean rho value
        libStatMap = {} # Output dict libSize key : list of ComputeError dicts
        libPredMap = {} # Output dict libSize key : list of predictions

        libRhoMap_self = {} # Output dict libSize key : mean rho value for self-prediction
        libStatMap_self = {} # Output dict libSize key : list of ComputeError dicts for self-prediction
        # libPredMap_self = {} # Output dict libSize key : list of self-predictions

        # colVec = self.Data[[S.columns[0]]].to_numpy()

        # Loop for library sizes
        # recon mission-JPL
        lib_picks_dict = {}
        lib_perf_dict = {}

        lib_perf_dict_self = {}
        for libSize in self.libSizes :
            # print(f'libSize: {libSize}')
            lib_picks_dict[libSize] = []
            # lib_perf_dict[libSize] = []
            rhos = zeros( self.sample )
            rhos_self = zeros( self.sample )

            # ## modification
            # col_preds = zeros( (len(S.pred_i), self.sample+1) )
            # tar_preds = zeros( (len(S.pred_i), self.sample+1) )

            if self.includeData :
                predictStats = [None] * self.sample
                predictStats_self = [None] * self.sample

            # Loop for subsamples
            for s in range( self.sample ) :
                # Generate library row indices for this subsample
                rng_i = RNG.choice( lib_i, size = min( libSize, N_lib_i ),
                                    replace = False )
                # trying to understand composition of library at higher lib sizes
                lib_picks_dict[libSize].append(rng_i)

                S.lib_i = rng_i
                S.knn = min([knn, len(S.lib_i) - 1])

                S.FindNeighbors() # Depends on S.lib_i
                # S.knn_neighbors is the set of knn nearest neighbor indexes for each prediction row (as determined in col_var space)

                # print(S.knn_neighbors.shape)
                # Code from Simplex:Project ---------------------------------
                # First column is minimum distance of all N pred rows
                minDistances = S.knn_distances[:,0]
                # In case there is 0 in minDistances: minWeight = 1E-6
                minDistances = fmax( minDistances, 1E-6 )

                # Divide each column of N x k knn_distances by minDistances
                scaledDistances = divide(S.knn_distances, minDistances[:,None])
                weights         = exp( -scaledDistances )  # Npred x k
                weightRowSum    = sum( weights, axis = 1 ) # Npred x 1

                # Matrix of knn_neighbors + Tp defines library target values
                knn_neighbors_Tp = S.knn_neighbors + self.Tp      # Npred x k
                # if predictant in the columns, we won't consider Tp as we will assume that is already accounted for
                if 'predictant' in S.columns:
                    knn_neighbors_Tp = S.knn_neighbors
                # knn_neighbors_Tp is the set of knn nearest neighbor prediction indexes (located at index +Tp)
                # for each prediction row
                libTargetValues = zeros( knn_neighbors_Tp.shape ) # Npred x k
                if S.selfPredict is not None:
                    libTargetvalues_self = zeros( knn_neighbors_Tp.shape ) # Npred x k

                for j in range( knn_neighbors_Tp.shape[1] ) :
                    libTargetValues[ :, j ][ :, None ] = \
                        S.targetVec[ knn_neighbors_Tp[ :, j ] ]

                    if S.selfPredict is not None:
                        libTargetvalues_self[ :, j ][ :, None ] = \
                            S.targetVec_self[ knn_neighbors_Tp[ :, j ] ]


                # Code from Simplex:Project ----------------------------------
                # print(libTargetValues.shape, libTargetValues, weights)
                # Projection is average of weighted knn library target values
                if self.weighted is True:
                    projection = sum( weights * libTargetValues,
                                  axis = 1) / weightRowSum
                else:
                    projection = sum(libTargetValues, axis=1) / S.knn

                    if S.selfPredict is not None:
                        projection_self = sum( libTargetvalues_self,
                                      axis = 1) / S.knn



                # Align observations & predictions as in FormatProjection()
                # Shift projection by Tp
                projection = roll( projection, S.Tp )
                if S.Tp > 0 :
                    projection[ :S.Tp ] = nan
                elif S.Tp < 0 :
                    projection[ S.Tp: ] = nan

                if S.selfPredict is not None:
                    projection_self = roll(projection_self, S.Tp)
                    if S.Tp > 0 :
                        projection_self[:S.Tp ] = nan
                    elif S.Tp < 0 :
                        projection_self[S.Tp:] = nan

                # calculate error based on predictions not made on library data
                # mask = in1d(S.pred_i, S.lib_i)
                # err = ComputeError(S.targetVec[S.pred_i[mask], 0],
                #                    projection[mask], digits=5)

                #
                # # TODO Remove the lib_i from the pred_i list
                if self.pred_num is not None:
                    RNG = default_rng(self.seed)
                    if self.pred_num <1:
                        pred_num = int(self.pred_num * (len(S.pred_i)-1))    # if pred_num is a fraction
                    else:
                        pred_num = int(self.pred_num)

                    pred_samplings = []
                    pred_samplings_self = []
                    available_proj_ind = where(~isnan(projection))[0]
                    for _ in range(30):
                        # pred_sample = RNG.choice(S.pred_i, size=min(len(S.pred_i)-1, pred_num),
                        #                        replace=False)
                        pred_sample = RNG.choice(available_proj_ind, size=min(len(available_proj_ind) - 1, pred_num),
                                                 replace=False)
                        # print('pred_sample', pred_sample)
                        # print('S.targetVec', type(S.targetVec), S.targetVec)
                        # print('projection', type(projection), projection)
                        # print('S.pred_i', S.pred_i)

                        err = ComputeError(S.targetVec[pred_sample, 0],
                                           projection[pred_sample], digits=5)

                        pred_samplings.append(err)

                        if S.selfPredict is not None:
                            err_self = ComputeError(S.targetVec_self[pred_sample, 0],
                                                    projection_self[pred_sample], digits=5)
                            pred_samplings_self.append(err_self)

                    pred_stats = DataFrame(pred_samplings)
                    err = pred_stats.mean(axis=0).to_dict()

                    if S.selfPredict is not None:
                        pred_stats_self = DataFrame(pred_samplings_self)
                        err_self = pred_stats_self.mean(axis=0).to_dict()

                        # bool_mask = isin(S.pred_i, pred_sample)
                        # bool_pred_saple = S.pred_i == pred_sample
                    # print('len(S.targetVec)', len(S.targetVec), len(S.targetVec[S.pred_i, 0 ]), len(bool_mask))
                    # print('len(projection)', len(projection))
                    # print(S.targetVec[ pred_sample,0], projection)
                else:
                    err = ComputeError( S.targetVec[S.pred_i, 0 ],
                                    projection, digits = 5 )

                    if S.selfPredict is not None:
                        err_self = ComputeError(S.targetVec_self[S.pred_i, 0],
                                                projection_self, digits=5)

                # tar_preds[:, 0] = S.targetVec[ S.pred_i, 0 ]
                # # print(S.lib_i)
                # # print(S.pred_i)
                # # print(S.pred_i)
                # # err['pred_i'] = S.pred_i

                rhos[ s ] = err['rho']
                if S.selfPredict is not None:
                    rhos_self[s] = err_self['rho']
                    lib_perf_dict_self[libSize] = rhos_self

                if self.includeData :
                    predictStats[s] = err
                    if S.selfPredict is not None:
                        predictStats_self[s] = err_self

                # ## modification
                # # Save the predictions for this subsample
                # tar_preds[ :, s+1 ] = projection
                #
                # libTargetValues = zeros(knn_neighbors_Tp.shape)  # Npred x k
                # for j in range(knn_neighbors_Tp.shape[1]):
                #     libTargetValues[:, j][:, None] = \
                #         colVec[knn_neighbors_Tp[:, j]]
                # # Code from Simplex:Project ----------------------------------
                # # print(libTargetValues.shape, libTargetValues, weights)
                # # Projection is average of weighted knn library target values
                # projection = sum(weights * libTargetValues,
                #                  axis=1) / weightRowSum
                #
                # # Align observations & predictions as in FormatProjection()
                # # Shift projection by Tp
                # projection = roll(projection, S.Tp)
                # if S.Tp > 0:
                #     projection[:S.Tp] = nan
                # elif S.Tp < 0:
                #     projection[S.Tp:] = nan
                #
                # col_preds[ :, s+1 ] = projection
                # col_preds[:, 0] = S.targetVec[S.pred_i, 0]



            def aggregate_data(data, func):
                return func(data)

            libRhoMap[ libSize ] = aggregate_data(rhos, self.aggMethod)
            lib_perf_dict[libSize]=rhos

            if S.selfPredict is not None:
                libRhoMap_self[ libSize ] = aggregate_data(rhos_self, self.aggMethod)
                lib_perf_dict_self[libSize] = rhos_self

            if self.includeData :
                libStatMap[ libSize ] = predictStats
                if S.selfPredict is not None:
                    libStatMap_self[libSize] = predictStats_self
                # libPredMap[ libSize ] = tar_preds

        # Reset S.lib_i to original
        S.lib_i = lib_i

        if self.includeData :
            return { 'columns' : S.columns, 'target' : S.target, 'lib_pics': lib_picks_dict, 'lib_perf':lib_perf_dict,
                     'libRho' : libRhoMap, 'predictStats' : libStatMap, 'predictions' : libPredMap, 'libRho_self': libRhoMap_self,
                     'predictStats_self': libStatMap_self }
        else :
            return {'columns':S.columns, 'target':S.target, 'libRho':libRhoMap}

    #--------------------------------------------------------------------
    def Validate( self ):
    #--------------------------------------------------------------------
        if self.verbose:
            print( f'{self.name}: Validate()' )

        if not len( self.libSizes ) :
            raise RuntimeError(f'{self.name} Validate(): LibSizes required.')
        if not IsIterable( self.libSizes ) :
            self.libSizes = [ int(L) for L in self.libSizes.split() ]

        if self.sample == 0:
            raise RuntimeError(f'{self.name} Validate(): ' +\
                               'sample must be non-zero.')

        # libSizes
        #   if 3 arguments presume [start, stop, increment]
        #      if increment < stop generate the library sequence.
        #      if increment > stop presume list of 3 library sizes.
        #   else: Already list of library sizes.
        if len( self.libSizes ) == 3 :
            # Presume ( start, stop, increment ) sequence arguments
            start, stop, increment = [ int( s ) for s in self.libSizes ]

            # If increment < stop, presume start : stop : increment
            # and generate the sequence of library sizes 
            if increment < stop :
                if increment < 1 :
                    msg = f'{self.name} Validate(): ' +\
                          f'libSizes increment {increment} is invalid.'
                    raise RuntimeError( msg )

                if start > stop :
                    msg = f'{self.name} Validate(): ' +\
                          f'libSizes start {start} stop {stop} are invalid.'
                    raise RuntimeError( msg )

                if start < self.E :
                    msg = f'{self.name} Validate(): ' +\
                          f'libSizes start {start} less than E {self.E}'
                    raise RuntimeError( msg )
                elif start < 3 :
                    msg = f'{self.name} Validate(): ' +\
                          f'libSizes start {start} less than 3.'
                    raise RuntimeError( msg )

                # Fill in libSizes sequence
                self.libSizes = [i for i in range(start, stop+1, increment)]

        if self.libSizes[-1] > self.Data.shape[0] :
            msg = f'{self.name} Validate(): ' +\
                  f'Maximum libSize {self.libSizes[-1]}'    +\
                  f' exceeds data size ({self.Data.shape[0]}).'
            raise RuntimeError( msg )

        if self.Tp < 0 :
            embedShift = abs( self.tau ) * ( self.E - 1 )
            maxLibSize = self.libSizes[-1]
            maxAllowed = self.Data.shape[0] - embedShift + (self.Tp + 1)
            if maxLibSize > maxAllowed :
                msg = f'{self.name} Validate(): Maximum libSize {maxLibSize}'  +\
                    f' too large for Tp {self.Tp}, E {self.E}, tau {self.tau}' +\
                    f' Maximum is {maxAllowed}'
                raise RuntimeError( msg )
