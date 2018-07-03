
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from models.model_library import get_model


class StackingModel(object):
    def __init__(self, base_estimators, params, logger, init_embedding_matrix=None):
        self.base_estimators = base_estimators
        self.params = params
        self.logger = logger
        self.init_embedding_matrix = init_embedding_matrix
        self.stacking_model = LogisticRegression()


    def fit(self, X_train, Q, validation_data, shuffle=True):
        # level1
        y_pred = np.zeros((validation_data["label"].shape[0], len(self.base_estimators)), dtype=np.float64)
        for i,base_estimator in enumerate(self.base_estimators):
            base_estimator.fit(X_train, Q, validation_data, shuffle)
            y_pred[:,i] = base_estimator.predict_proba(validation_data, Q).flatten()
            self.logger.info("%s logloss: %.5f" % (base_estimator.model_name, log_loss(validation_data["label"], y_pred[:,i])))

        # level2
        self.stacking_model.fit(y_pred, validation_data["label"])
        p = self.stacking_model.predict_proba(y_pred)[:,1].flatten()
        self.logger.info("stacking logloss: %.5f"%log_loss(validation_data["label"], p))
        return self


    def predict_proba(self, X, Q):
        # level1
        y_pred = np.zeros((X["label"].shape[0], len(self.base_estimators)), dtype=np.float64)
        for i,base_estimator in enumerate(self.base_estimators):
            y_pred[:,i] = base_estimator.predict_proba(X, Q).flatten()
        # level2
        return self.stacking_model.predict_proba(y_pred)[:,1].flatten()


    def fit_predict_proba(self, X_train, Q, validation_data, test_data, shuffle=True, refit_valid_epoch=5):
        # level1
        y_pred_valid = np.zeros((validation_data["label"].shape[0], len(self.base_estimators)), dtype=np.float64)
        y_pred_test = np.zeros((test_data["label"].shape[0], len(self.base_estimators)), dtype=np.float64)
        for i,model_type in enumerate(self.base_estimators):
            base_estimator = get_model(model_type)(self.params, self.logger, init_embedding_matrix=self.init_embedding_matrix)
            base_estimator.fit(X_train, Q, validation_data, shuffle)
            y_pred_valid[:,i] = base_estimator.predict_proba(validation_data, Q).flatten()
            self.logger.info("%s logloss: %.5f" % (model_type, log_loss(validation_data["label"], y_pred_valid[:, i])))
            # fit a few more epoch on validation data
            base_estimator.fit(validation_data, Q, None, shuffle, total_epoch=refit_valid_epoch)
            y_pred_test[:,i] = base_estimator.predict_proba(test_data, Q).flatten()


        # level2
        self.stacking_model.fit(y_pred_valid, validation_data["label"])
        p_valid = self.stacking_model.predict_proba(y_pred_valid)[:,1].flatten()
        p_test = self.stacking_model.predict_proba(y_pred_test)[:,1].flatten()
        self.logger.info("stacking logloss: %.5f"%log_loss(validation_data["label"], p_valid))
        return p_test
