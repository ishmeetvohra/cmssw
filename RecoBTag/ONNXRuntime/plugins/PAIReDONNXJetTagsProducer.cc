#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include "RecoBTag/FeatureTools/interface/SecondaryVertexConverter.h"
#include "RecoBTag/FeatureTools/interface/NeutralCandidateConverter.h"
#include "RecoBTag/FeatureTools/interface/ChargedCandidateConverter.h"
#include "RecoBTag/FeatureTools/interface/paired_helper.h"

#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"

#include "Rivet/Tools/ParticleIdUtils.hh"

using namespace cms::Ort;
using namespace btagbtvdeep;
using namespace Rivet;

class PAIReDONNXJetTagsProducer : public edm::stream::EDProducer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit PAIReDONNXJetTagsProducer(const edm::ParameterSet&, const ONNXRuntime*);
  ~PAIReDONNXJetTagsProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const ONNXRuntime*);

private:
  typedef reco::VertexCompositePtrCandidateCollection SVCollection;
  typedef reco::VertexCollection VertexCollection;

  const std::string name_, name_pf_, name_sv_;
  const edm::EDGetTokenT<edm::View<pat::Jet>> jet_token_;
  edm::EDGetTokenT<reco::CandidateView> cand_token_;
  edm::EDGetTokenT<edm::View<reco::GenParticle>> gen_particle_token_;
  edm::EDGetTokenT<VertexCollection> vtx_token_;
  const edm::EDGetTokenT<SVCollection> sv_token_;
  // define producer functions
  void beginStream(edm::StreamID) override {}
  void produce(edm::Event&, const edm::EventSetup&) override;
  std::vector<size_t> sort_pf_cands(edm::Event& iEvent,
                                    const edm::EventSetup& iSetup,
                                    edm::Handle<edm::View<reco::Candidate>> cands);
  SVCollection sort_svs(edm::Event& iEvent,
                        const edm::EventSetup& iSetup,
                        edm::Handle<SVCollection> svs,
                        edm::Handle<VertexCollection> vtxs);
  void make_inputs(edm::Event& iEvent,
                   const edm::EventSetup& iSetup,
                   edm::Handle<edm::View<pat::Jet>> jets,
                   unsigned i_jet,
                   unsigned j_jet,
                   SVCollection svs,
                   edm::Handle<edm::View<reco::Candidate>> cands,
                   edm::Handle<VertexCollection> vtxs,
                   std::vector<size_t> pf_sorted_idx);
  int get_n_parton(edm::Event& iEvent,
                   const edm::EventSetup& iSetup,
                   edm::Handle<edm::View<pat::Jet>> jets,
                   unsigned i_jet,
                   unsigned j_jet,
                   edm::Handle<edm::View<reco::GenParticle>> gen_particles,
                   int parton_id);
  void endStream() override {}
  // store hard-coded constants indicating size, structure, and names of input arrays
  enum InputIndexes {
    kPfCandFeatures = 0,
    kPfCandVectors = 1,
    kSVFeatures = 2,
    kSVVectors = 3
  };
  constexpr static unsigned n_features_vector_ = 4;
  constexpr static unsigned n_features_pf_ = 16;
  constexpr static unsigned n_features_sv_ = 17;
  unsigned max_pf_cands = 128;
  unsigned max_svs = 10;
  std::vector<std::vector<int64_t>> input_shapes_ = {{(int64_t)1, (int64_t)max_pf_cands, (int64_t)n_features_pf_},
                                                     {(int64_t)1, (int64_t)max_pf_cands, (int64_t)n_features_vector_},
                                                     {(int64_t)1, (int64_t)max_svs, (int64_t)n_features_sv_},
                                                     {(int64_t)1, (int64_t)max_svs, (int64_t)n_features_vector_}};
  FloatArrays data_;  // initialize actual input array
  std::vector<unsigned> input_sizes_ = {max_pf_cands * n_features_pf_,
                                        max_pf_cands * n_features_vector_,
                                        max_svs * n_features_sv_,
                                        max_svs * n_features_vector_};
  std::vector<std::string> input_names = {"inp1", "inp2", "inp3", "inp4"};//{"cpf", "cpf_4v", "sv", "sv_4v"};
  std::vector<std::string> output_names = {"scores"};
  
  // define loose cuts on jets to save space
  float jet_pt_cut = 20;  // uses raw pt
  float jet_eta_cut = 2.5;
};


// reads parameters from the python config
// declares which event data this module will read
// no actual reading yet only declaring the dependencies
PAIReDONNXJetTagsProducer::PAIReDONNXJetTagsProducer(const edm::ParameterSet& iConfig, const ONNXRuntime* cache)
    : name_(iConfig.getParameter<std::string>("name")),         // instance label of the NanoAOD table
      name_pf_(iConfig.getParameter<std::string>("name_pf")),
      name_sv_(iConfig.getParameter<std::string>("name_sv")),
      jet_token_(consumes<edm::View<pat::Jet>>(iConfig.getParameter<edm::InputTag>("jets"))), // reads the InputTag and stores the returned token
      cand_token_(consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("candidates"))),
      gen_particle_token_(consumes<edm::View<reco::GenParticle>>(iConfig.getParameter<edm::InputTag>("gen_particles"))),
      vtx_token_(consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      sv_token_(consumes<SVCollection>(iConfig.getParameter<edm::InputTag>("secondary_vertices"))) {
  produces<nanoaod::FlatTable>(name_); // declares what this module will produce
}

void PAIReDONNXJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("name", "PAIReDJets");
  desc.add<std::string>("name_pf", "PAIReDPF");
  desc.add<std::string>("name_sv", "PAIReDSV");
  desc.add<edm::InputTag>("jets", edm::InputTag("slimmedJetsPuppi"));
  desc.add<edm::InputTag>("candidates", edm::InputTag("packedPFCandidates"));
  desc.add<edm::InputTag>("gen_particles", edm::InputTag("prunedGenParticles"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlineSlimmedPrimaryVertices"));
  desc.add<edm::InputTag>("secondary_vertices", edm::InputTag("slimmedSecondaryVertices"));
  desc.add<edm::FileInPath>("model_path", edm::FileInPath("RecoBTag/Combined/data/PAIReD/PAIReD_6class_ak4_clustered/best_model_opset20_9.onnx")); // put my onnx model in RecoBTag/Combined/data/PAIReD
  descriptions.addWithDefaultLabel(desc);
  //descriptions.add("PAIReDJetTable", desc);
}

// load model once
std::unique_ptr<ONNXRuntime> PAIReDONNXJetTagsProducer::initializeGlobalCache(const edm::ParameterSet& iConfig) {
  return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("model_path").fullPath());
}

void PAIReDONNXJetTagsProducer::globalEndJob(const ONNXRuntime* cache) {}


// runs once per event: build input, run onnx for each jet pair, store scores and write NanoAOD table
void PAIReDONNXJetTagsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::vector<unsigned> input_sizes = { max_pf_cands * n_features_pf_,
                                        max_pf_cands * n_features_vector_,
                                        max_svs * n_features_sv_,
                                        max_svs * n_features_vector_};

  // changed this
  data_.clear();
  data_.reserve(input_sizes.size());

  for (auto len : input_sizes) {
    data_.emplace_back(len, 0.f);
  }

  // initialize output variables: per row columns of the output NanoAOD table
  std::vector<int> n_bparton, n_cparton;
  std::vector<unsigned> idx_jet1, idx_jet2;
  std::vector<float> score_BB, score_CC, score_bb, score_bl, score_cl, score_ll;
  std::vector<float> outputs;
  //std::vector<float> reg_pt, reg_mass, reg_phi, reg_eta; // for future when jet mass and kinematics are regressed from the model
  // initialize event components
  edm::Handle<reco::VertexCollection> vtxs;
  iEvent.getByToken(vtx_token_, vtxs);
  edm::Handle<edm::View<pat::Jet>> jets;
  iEvent.getByToken(jet_token_, jets);
  edm::Handle<edm::View<reco::Candidate>> cands;
  iEvent.getByToken(cand_token_, cands);
  edm::Handle<edm::View<reco::GenParticle>> gen_particles;
  iEvent.getByToken(gen_particle_token_, gen_particles);
  edm::Handle<reco::VertexCompositePtrCandidateCollection> svs;
  iEvent.getByToken(sv_token_, svs);
  // sort pf cands and svs
  float num_valid = 0;
  std::vector<size_t> pf_sorted_idx = sort_pf_cands(iEvent, iSetup, cands);
  reco::VertexCompositePtrCandidateCollection svs_sorted = sort_svs(iEvent, iSetup, svs, vtxs);
  bool isMC = !(iEvent.isRealData());  // store MCvsData boolean for adding truth information
  // loop over paired jets and institute pt/eta cuts
  for (unsigned i_jet = 0; i_jet < jets->size() - 1 && jets->size() > 0; ++i_jet) {
    const auto& jet1 = jets->at(i_jet);
    if (jet1.pt() * jet1.jecFactor("Uncorrected") < jet_pt_cut || abs(jet1.eta()) > jet_eta_cut)
      continue;
    for (unsigned j_jet = i_jet + 1; j_jet < jets->size(); ++j_jet) {
      //if (num_valid > 0) break;
      const auto& jet2 = jets->at(j_jet);
      if (jet2.pt() * jet2.jecFactor("Uncorrected") < jet_pt_cut || abs(jet2.eta()) > jet_eta_cut)
        continue;
      // init input data storage and fill it
      //std::cout<<"jets "<<i_jet<<" and "<<j_jet<<std::endl;
      make_inputs(iEvent, iSetup, jets, i_jet, j_jet, svs_sorted, cands, vtxs, pf_sorted_idx); // build inputs and run Onnx inference for each jet pair
      outputs = globalCache()->run(input_names, data_, input_shapes_)[0]; // output
      assert(outputs.size() == 6);  // 6 classes
      num_valid += 1;
      idx_jet1.emplace_back(i_jet);
      idx_jet2.emplace_back(j_jet);
      score_BB.push_back(outputs[0]);
      score_CC.push_back(outputs[1]);
      score_bb.push_back(outputs[2]);
      score_bl.push_back(outputs[3]);
      score_cl.push_back(outputs[4]);
      score_ll.push_back(outputs[5]);
      // reg_pt.emplace_back(outputs[3]);
      // reg_mass.emplace_back(outputs[4]);
      // reg_eta.emplace_back(outputs[5]);
      // reg_phi.emplace_back(outputs[6]);
      if (isMC) {
        n_bparton.emplace_back(get_n_parton(iEvent, iSetup, jets, i_jet, j_jet, gen_particles, 5));
        n_cparton.emplace_back(get_n_parton(iEvent, iSetup, jets, i_jet, j_jet, gen_particles, 4));
      }
    }
  }
  // save paired jet flat table
  auto pjTable = std::make_unique<nanoaod::FlatTable>(idx_jet1.size(), name_, false);
  pjTable->addColumn<unsigned>("idx_jet1", idx_jet1, "Index of constituent jet 1");
  pjTable->addColumn<unsigned>("idx_jet2", idx_jet2, "Index of constituent jet 2");
  pjTable->addColumn<float>("BB_score", score_BB, "Model score for BB jet", 10);
  pjTable->addColumn<float>("CC_score", score_CC, "Model score for CC jet", 10);
  pjTable->addColumn<float>("bb_score", score_bb, "Model score for bb jet", 10);
  pjTable->addColumn<float>("bl_score", score_bl, "Model score for bl jet", 10);
  pjTable->addColumn<float>("cl_score", score_cl, "Model score for cl jet", 10);
  pjTable->addColumn<float>("ll_score", score_ll, "Model score for ll jet", 10);
  // pjTable->addColumn<float>("mass", reg_mass, "Regressed PAIReD jet mass", 10);
  // pjTable->addColumn<float>("pt", reg_pt, "Regressed PAIReD jet pt", 10);
  // pjTable->addColumn<float>("eta", reg_eta, "Regressed PAIReD jet eta", 10);
  // pjTable->addColumn<float>("phi", reg_phi, "Regressed PAIReD jet phi", 10);
  if (isMC) {
    pjTable->addColumn<int>("n_bparton", n_bparton, "Number of b partons", 10);
    pjTable->addColumn<int>("n_cparton", n_cparton, "Number of c partons", 10);
  }
  iEvent.put(std::move(pjTable), name_);
}

std::vector<size_t> PAIReDONNXJetTagsProducer::sort_pf_cands(edm::Event& iEvent,
                                                             const edm::EventSetup& iSetup,
                                                             edm::Handle<edm::View<reco::Candidate>> cands) {
  // sort pf candidates with positive PUPPI weight by pt
  std::vector<float> pf_pts;
  std::vector<size_t> pf_unsorted_idx;
  // generate list of indices and pts with positive puppi weight
  for (unsigned entry = 0; entry < cands->size(); ++entry) {
    const reco::Candidate* cand = &(cands->at(entry));
    auto packed_cand = dynamic_cast<const pat::PackedCandidate*>(cand);
    if (packed_cand->puppiWeight() > 0) {
      pf_pts.emplace_back(cand->pt());
      pf_unsorted_idx.emplace_back(entry);
    }
  }
  // sort the list of pts
  if (pf_unsorted_idx.size() < 2)
    return pf_unsorted_idx;
  std::vector<size_t> pf_sorted_idx(pf_unsorted_idx.size());
  std::iota(pf_sorted_idx.begin(), pf_sorted_idx.end(), 0);
  std::sort(
      pf_sorted_idx.begin(), pf_sorted_idx.end(), [&pf_pts](size_t i, size_t j) { return pf_pts[i] > pf_pts[j]; });
  // put back original indices into sorted list
  for (unsigned i = 0; i < pf_unsorted_idx.size(); ++i) {
    pf_sorted_idx[i] = pf_unsorted_idx[pf_sorted_idx[i]];
  }
  return pf_sorted_idx;
}

reco::VertexCompositePtrCandidateCollection PAIReDONNXJetTagsProducer::sort_svs(edm::Event& iEvent,
                                                                                const edm::EventSetup& iSetup,
                                                                                edm::Handle<SVCollection> svs,
                                                                                edm::Handle<VertexCollection> vtxs) {
  // sort secondary vertices by dxy
  const auto& pv = vtxs->at(0);
  auto svs_sorted = *svs;
  std::sort(svs_sorted.begin(), svs_sorted.end(), [&pv](const auto& sva, const auto& svb) {
    return vertexD3d(sva, pv).value() / vertexD3d(sva, pv).error() >
           vertexD3d(svb, pv).value() / vertexD3d(svb, pv).error();
  });
  return svs_sorted;
}

void PAIReDONNXJetTagsProducer::make_inputs(edm::Event& iEvent,
                                            const edm::EventSetup& iSetup,
                                            edm::Handle<edm::View<pat::Jet>> jets,
                                            unsigned i_jet,
                                            unsigned j_jet,
                                            SVCollection svs,
                                            edm::Handle<edm::View<reco::Candidate>> cands,
                                            edm::Handle<VertexCollection> vtxs,
                                            std::vector<size_t> pf_sorted_idx) {
  // get primary vertex and jet data
  const auto& pv = vtxs->at(0);
  const auto& jet1 = jets->at(i_jet);
  edm::RefToBase<pat::Jet> jet_ref1(jets, i_jet);
  const auto& jet2 = jets->at(j_jet);
  edm::RefToBase<pat::Jet> jet_ref2(jets, j_jet);
  float jet1_pt = jet1.pt() * jet1.jecFactor("Uncorrected");
  float jet2_pt = jet2.pt() * jet2.jecFactor("Uncorrected");
  float jet1_energy = jet1.energy() * jet1.jecFactor("Uncorrected");
  float jet2_energy = jet2.energy() * jet2.jecFactor("Uncorrected");
  // initialize data vector
  for (unsigned igroup = 0; igroup < input_names.size(); ++igroup) {
    auto &group_values = data_[igroup];
    group_values.resize(input_sizes_[igroup]);
    std::fill(group_values.begin(), group_values.end(), 0);
  }
  //find valid pf cands and save indices
  std::vector<size_t> pf_idx;
  for (size_t entry_i = 0; pf_idx.size() < max_pf_cands && entry_i < pf_sorted_idx.size(); ++entry_i) {
    auto entry = pf_sorted_idx[entry_i];
    const reco::Candidate* cand = &(cands->at(entry));
    if (false && inEllipse(jet1.eta(), jet1.phi(), jet2.eta(), jet2.phi(), (*cand).eta(), (*cand).phi())) {
      pf_idx.emplace_back(entry); // ellipse case does never happen because of the false
      //std::cout<<"new entry "<<entry_i<<" pt "<<(*cand).pt()<<std::endl;
    } else if (true && (isClusteredToJet(cand, jet1) || isClusteredToJet(cand, jet2))) {
      pf_idx.emplace_back(entry); // deleted: (*cand).pt() > 500 because not included in training
      //std::cout<<"new entry "<<entry_i<<" pt "<<(*cand).pt()<<std::endl;
    } else {
      continue;
    }
  }
  
  //find valid secondary vertices and save indices
  std::vector<size_t> sv_idx;
  const float R = 0.4;

  for (size_t sv_i = 0; sv_i < svs.size() && sv_idx.size() < max_svs; ++sv_i) {
    const auto& sv = svs.at(sv_i);

    //keep SV if within AK4 radius of either jet
    if (reco::deltaR(sv, jet1) > R && reco::deltaR(sv, jet2) > R) continue;

    sv_idx.emplace_back(sv_i);
  }



  auto &cpf  = data_[kPfCandFeatures];
  auto &cpf4 = data_[kPfCandVectors];
  auto &svf = data_[kSVFeatures];
  auto &sv4 = data_[kSVVectors];

  //cpf and cpf4 are already padded fixed length
  size_t out_i = 0;
  const size_t n_pf = std::min(pf_idx.size(), (size_t)max_pf_cands);

  for (size_t i = 0; i < n_pf && out_i < max_pf_cands; ++i) {
    const auto entry = pf_idx[i];
    const reco::Candidate* cand = &(cands->at(entry));

    const auto* packed = dynamic_cast<const pat::PackedCandidate*>(cand);
    if (!packed) continue;

    //cpf candidates
    const size_t base = out_i * n_features_pf_;

    cpf[base + 0]  = catch_infs_and_bound((std::log(packed->pt()) - 1.0f) * 0.25f, 0, -5, 5);  // part_pt_log
    cpf[base + 1]  = catch_infs_and_bound((std::log(packed->energy()) - 2.0f) * 0.25f, 0, -5, 5); // part_e_log
    cpf[base + 2]  = catch_infs_and_bound((std::log(packed->pt() / (jet1_pt + jet2_pt)) + 4.7f) * 0.25f, 0, -5, 5); // part_logptrel
    cpf[base + 3]  = catch_infs_and_bound((std::log(packed->energy() / (jet1_energy + jet2_energy)) + 4.7f) * 0.25f, 0, -5, 5); // part_logerel
    cpf[base + 4]  = catch_infs_and_bound(reco::deltaR(*packed, jet1), 0, -5, 5); // part_deltaR1
    cpf[base + 5]  = catch_infs_and_bound(reco::deltaR(*packed, jet2), 0, -5, 5); // part_deltaR2
    cpf[base + 6]  = catch_infs_and_bound((float)packed->charge(), 0, -1e32, 1e32); // part_charge

    // track-dependent quantities
    if (packed->charge() != 0 && packed->hasTrackDetails()) {
      cpf[base + 7]  = catch_infs_and_bound(std::tanh(packed->dxy()), 0, -1e32, 1e32); // part_d0
      cpf[base + 8]  = catch_infs_and_bound((float)packed->dxyError(), 0, 0, 1);       // part_d0err
      cpf[base + 9]  = catch_infs_and_bound(std::tanh(packed->dz()), 0, -1e32, 1e32);  // part_dz
      cpf[base + 10] = catch_infs_and_bound((float)packed->dzError(), 0, 0, 1);        // part_dzerr
    } else {
      cpf[base + 7]  = std::tanh(-1.f);
      cpf[base + 8]  = 0.f;
      cpf[base + 9]  = std::tanh(-1.f);
      cpf[base + 10] = 0.f;
    }

    // deta sign convention from training: (part_eta - jet_eta) * (-1)**(jet_eta < 0)
    const float sign1 = (jet1.eta() < 0.f) ? -1.f : 1.f;
    const float sign2 = (jet2.eta() < 0.f) ? -1.f : 1.f;

    cpf[base + 11] = catch_infs_and_bound((packed->eta() - jet1.eta()) * sign1, 0, -1e32, 1e32); // part_deta1
    cpf[base + 12] = catch_infs_and_bound(reco::deltaPhi(packed->phi(), jet1.phi()), 0, -1e32, 1e32); // part_dphi1
    cpf[base + 13] = catch_infs_and_bound((packed->eta() - jet2.eta()) * sign2, 0, -1e32, 1e32); // part_deta2
    cpf[base + 14] = catch_infs_and_bound(reco::deltaPhi(packed->phi(), jet2.phi()), 0, -1e32, 1e32); // part_dphi2
    cpf[base + 15] = catch_infs_and_bound((float)packed->puppiWeight(), 0, -1e32, 1e32); // part_puppiweight

    //cpf vectors
    const size_t base4 = out_i * n_features_vector_;
    cpf4[base4 + 0] = catch_infs_and_bound((float)packed->px(), 0, -1e32, 1e32);
    cpf4[base4 + 1] = catch_infs_and_bound((float)packed->py(), 0, -1e32, 1e32);
    cpf4[base4 + 2] = catch_infs_and_bound((float)packed->pz(), 0, -1e32, 1e32);
    cpf4[base4 + 3] = catch_infs_and_bound((float)packed->energy(), 0, -1e32, 1e32);

    ++out_i;
  }


  size_t out_sv = 0;
  const size_t n_sv = std::min(sv_idx.size(), (size_t)max_svs);

  for (size_t i = 0; i < n_sv && out_sv < max_svs; ++i) {
    const auto& sv = svs.at(sv_idx[i]);

    //sv_features
    const size_t base = out_sv * n_features_sv_;

    //sv_charge: sum of daughter charges
    float sv_charge = 0.f;
    for (size_t k = 0; k < sv.numberOfDaughters(); ++k) {
      sv_charge += sv.daughter(k)->charge();
    }

    const auto& d3d = vertexD3d(sv, pv);
    const auto& dxy = vertexDxy(sv, pv);

    svf[base + 0]  = catch_infs_and_bound(sv_charge, 0, -1e32, 1e32);               // sv_charge
    svf[base + 1]  = catch_infs_and_bound(sv.vertexNormalizedChi2() * 0.25f, 0, -5, 5); // sv_chi2
    svf[base + 2]  = catch_infs_and_bound(d3d.value() * 0.05f, 0, -5, 5);             // sv_dlen
    svf[base + 3]  = catch_infs_and_bound((d3d.value() / d3d.error()) * 0.01f, 0, -5, 5); // sv_dlenSig
    svf[base + 4]  = catch_infs_and_bound(dxy.value() * 0.05f, 0, -5, 5);             // sv_dxy
    svf[base + 5]  = catch_infs_and_bound((dxy.value() / dxy.error()) * 0.01f, 0, -5, 5); // sv_dxySig
    svf[base + 6]  = catch_infs_and_bound(sv.eta(), 0, -1e32, 1e32);                  // sv_eta
    svf[base + 7]  = catch_infs_and_bound(sv.mass() * 0.1f, 0, -5, 5);                // sv_mass
    svf[base + 8]  = catch_infs_and_bound(sv.vertexNdof() * 0.1f, 0, -5, 5);          // sv_ndof
    svf[base + 9]  = catch_infs_and_bound((float)sv.numberOfDaughters() * 0.1f, 0, -1e32, 1e32); // sv_ntracks
    svf[base + 10] = catch_infs_and_bound(std::acos(-vertexDdotP(sv, pv)) * 0.33f, 0, -5, 5); // sv_pAngle
    svf[base + 11] = catch_infs_and_bound(sv.phi(), 0, -1e32, 1e32);                  // sv_phi
    svf[base + 12] = catch_infs_and_bound(sv.pt() * 0.01f, 0, -5, 5);                 // sv_pt

    // deta / dphi with signed convention 
    const float sign1 = (jet1.eta() < 0.f) ? -1.f : 1.f;
    const float sign2 = (jet2.eta() < 0.f) ? -1.f : 1.f;

    svf[base + 13] = catch_infs_and_bound((sv.eta() - jet1.eta()) * sign1, 0, -1e32, 1e32); // sv_deta1
    svf[base + 14] = catch_infs_and_bound((sv.eta() - jet2.eta()) * sign2, 0, -1e32, 1e32); // sv_deta2
    svf[base + 15] = catch_infs_and_bound(reco::deltaPhi(sv.phi(), jet1.phi()), 0, -1e32, 1e32); // sv_dphi1
    svf[base + 16] = catch_infs_and_bound(reco::deltaPhi(sv.phi(), jet2.phi()), 0, -1e32, 1e32); // sv_dphi2

    //sv_vectors
    const size_t base4 = out_sv * n_features_vector_;
    sv4[base4 + 0] = catch_infs_and_bound((float)sv.px(), 0, -1e32, 1e32);
    sv4[base4 + 1] = catch_infs_and_bound((float)sv.py(), 0, -1e32, 1e32);
    sv4[base4 + 2] = catch_infs_and_bound((float)sv.pz(), 0, -1e32, 1e32);
    sv4[base4 + 3] = catch_infs_and_bound((float)sv.energy(), 0, -1e32, 1e32);

    ++out_sv;
  }
 
}

#include "DataFormats/Math/interface/deltaR.h"  // reco::deltaR
// same as in getMCInfo used for truths in Training
int PAIReDONNXJetTagsProducer::get_n_parton(edm::Event& iEvent,
                                           const edm::EventSetup& iSetup,
                                           edm::Handle<edm::View<pat::Jet>> jets,
                                           unsigned i_jet,
                                           unsigned j_jet,
                                           edm::Handle<edm::View<reco::GenParticle>> gen_particles,
                                           int parton_id) {
  const auto& jet1 = jets->at(i_jet);
  const auto& jet2 = jets->at(j_jet);

  constexpr float R = 0.4f;

  int n_parton = 0;
  for (unsigned i = 0; i < gen_particles->size(); ++i) {
    const auto* genp = &(gen_particles->at(i));

    //last copy 
    if (!genp->isLastCopy()) continue;

    //reject quarks with no mother (pileup)
    if (genp->numberOfMothers() == 0 || genp->mother(0) == nullptr) continue;

    //within R of either jet
    const bool in_region =
        (reco::deltaR(*genp, jet1) < R) || (reco::deltaR(*genp, jet2) < R);
    if (!in_region) continue;

    //count charm and bottom
    if (parton_id == 4) {
      n_parton += Rivet::PID::hasCharm(genp->pdgId());
    } else if (parton_id == 5) {
      n_parton += Rivet::PID::hasBottom(genp->pdgId());
    }
  }
  return n_parton;
}

DEFINE_FWK_MODULE(PAIReDONNXJetTagsProducer);
