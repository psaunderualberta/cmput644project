class PopulationStorage:
    __abc_err_msg = "This is an abstract class"

    def insert_heuristic_if_better(self, h, f):
        raise NotImplementedError(self.__abc_err_msg)
    
    def get_random_mutated_heuristic(self, fitnesses):
        raise NotImplementedError(self.__abc_err_msg)
    
    def get_fitnesses(self, idx=None):
        raise NotImplementedError(self.__abc_err_msg)        
    
    def get_stored_data(self,  h, f):
        raise NotImplementedError(self.__abc_err_msg)
