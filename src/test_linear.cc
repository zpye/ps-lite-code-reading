#include "ps/ps.h"
#include <iostream>
using namespace ps;

class KVServerLinear {
public:
  KVServerLinear() {
    using namespace std::placeholders;
    ps_server = new KVServer< float >(0);
    ps_server->set_request_handle(std::bind(&KVServerLinear::KVServerLinearHandle, this, _1, _2, _3));

    learning_rate = 0.01f;
  }

  ~KVServerLinear() {
    if (ps_server) {
      delete ps_server;
    }
  }

private:
  void KVServerLinearHandle(const KVMeta& req_meta, 
                            const KVPairs< float >& req_data, 
                            KVServer< float >* server) {
    size_t n = req_data.keys.size();
    KVPairs< float > res;
    if (req_meta.push) {
      CHECK_EQ(n, req_data.vals.size());
    } else {
      res.keys = req_data.keys; 
      res.vals.resize(n);
    }

    if(store.empty()) {
      for(size_t i = 0; i < n; ++i) {
        Key key = req_data.keys[i];
        store[key] = req_data.vals[i];
      }
    } else {
      for (size_t i = 0; i < n; ++i) {
        Key key = req_data.keys[i];
        if (req_meta.push) {
          store[key] -= learning_rate * req_data.vals[i];
        } else {
          res.vals[i] = store[key];
        }
      }
    }

    server->Response(req_meta, res);
  }

  float learning_rate;
 
  std::unordered_map< Key, float > store;
  KVServer< float >* ps_server;
};


void StartServer() {
  if (!IsServer()) {
    return;
  }

  auto server = new KVServerLinear();
  RegisterExitCallback([server](){ delete server; });
}

void RunWorker() {
  if (!IsWorker()) return;
  KVWorker< float > kv(0, 0);

  // init rand
  srand(0);

  // init
  int dim = 10;
  std::vector< Key > keys(dim);
  std::vector< float > vals(dim);
  std::vector< float > init_vals(dim, 0);

  // linear parameters
  for (int i = 0; i < dim; ++i) {
    keys[i] = i;
    vals[i] = (float(rand() % 100) - 50.0f) / 10.0f;
  }

  for(int i = 0; i < dim; ++i) {
    std::cout << "real: " << vals[i] << std::endl;
  }

  // generate train data
  int rank = MyRank();
  srand(rank + 7);

  int num = 200;
  std::vector< std::vector< float > > data;
  for(int i = 0; i < num; ++i) {
    std::vector< float > d;
    
    float temp = 0.0f;
    for(int j = 0; j < dim; ++j) {
      float t = (float(rand() % 100) - 50.0f) / 10.0f;
      d.push_back(t);
      temp += t * vals[j];
    }

    d.push_back(temp);

    data.push_back(d);

    std::cout << "generate: " << i << std::endl;
  }

  // set parameters to server
  if(rank == 0) {
    kv.Wait(kv.Push(keys, init_vals));
  }

  Postoffice::Get()->Barrier(0, kWorkerGroup);

  // train
  for(int iter = 0; iter < 100; ++iter) {
    // pull
    std::vector< float > rets;
    kv.Wait(kv.Pull(keys, &rets));

    for(int i = 0; i < dim; ++i) {
      std::cout << rank << " " << iter << " " << keys[i] << " learn: " << rets[i] << std::endl;
    }

    // L2-norm
    std::vector< float > grad(dim, 0.0f);
    for(int i = 0; i < num; ++i) {
      float pred = 0.0f;
      for(int j = 0; j < dim; ++j) {
        pred += data[i][j] * rets[j]; 
      }
      for(int j = 0; j < dim; ++j) {
        grad[j] += (pred - data[i][dim]) * data[i][j];
      }
    } 

    for(int i = 0; i < dim; ++i) {
      grad[i] /= num;
    }

    // push
    std::cout << "Push" << std::endl;
    kv.Wait(kv.Push(keys, grad));
  }

  // pull
  std::vector< float > rets;
  kv.Wait(kv.Pull(keys, &rets));
  
  for(int i = 0; i < dim; ++i) {
    std::cout << rank << " real: " << vals[i] << " learn: " << rets[i] << std::endl;
  }
}

int main(int argc, char *argv[]) {
  // start system
  Start(0);

  // setup server nodes
  StartServer();
  
  // run worker nodes
  RunWorker();
  
  // stop system
  Finalize(0, true);
  
  return 0;
}
