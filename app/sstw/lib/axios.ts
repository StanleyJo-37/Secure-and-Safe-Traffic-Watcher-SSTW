import depAxios from "axios";

const axios = depAxios.create({
  baseURL: "http://localhost:5000",
});

axios.interceptors.request.use(
  async (resp) => {
    return resp;
  },
  async (err) => {}
);

axios.interceptors.response.use(
  (response) => response,
  (error) => {
    return Promise.reject(error);
  }
);

export default axios;
