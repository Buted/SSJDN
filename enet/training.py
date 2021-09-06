import os

from torchtext.data import BucketIterator

from enet.util import run_over_data
import time


def train(model, train_set, dev_set, test_set, optimizer_constructor, epochs, tester, parser):
    # build batch on cpu
    train_iter = BucketIterator(train_set, batch_size=parser.batch, train=False, shuffle=True, device=parser.device,
                                sort_key=lambda x: x.WORDS[0])
    dev_iter = BucketIterator(dev_set, batch_size=parser.batch, train=False, shuffle=True, device=parser.device,
                              sort_key=lambda x: len(x.WORDS))
    test_iter = BucketIterator(test_set, batch_size=parser.batch, train=False, shuffle=True, device=parser.device,
                               sort_key=lambda x: len(x.WORDS))
    scores = 0.0
    best_test_scores = 0.0
    best_scores = 0.0
    best_epoch = 0
    now_bad = 0
    restart_used = 0
    print("\nStarting training...\n")
    lr = parser.lr
    optimizer = optimizer_constructor(lr=lr)

    for i in range(epochs):
        parser.writer.add_scalar('train/lr', lr, i)
        # Training Phrase
        print("Epoch", i + 1)
        start = time.time()
        training_loss, training_ed_f1, training_ae_f1 = run_over_data(data_iter=train_iter,
                                                                     optimizer=optimizer,
                                                                     model=model,
                                                                     need_backward=True,
                                                                     tester=tester,
                                                                     device=model.device,
                                                                     maxnorm=parser.maxnorm,
                                                                     label_i2s=parser.label_i2s,
                                                                     save_output=False,
                                                                     seq_lambda=parser.seq_lambda,
                                                                     word_count_emb=parser.word_count_emb,
                                                                     word_num_emb=parser.word_num_emb
                                                                      )
        print('train an epoch:', time.time() - start)
        print("\nEpoch", i + 1, " training loss: ", training_loss,
              " training ed f1: ", training_ed_f1,
              " training ae f1: ", training_ae_f1)

        parser.writer.add_scalar('train/loss', training_loss, i)
        parser.writer.add_scalar('train/ed/f1', training_ed_f1, i)
        parser.writer.add_scalar('train/ae/f1', training_ae_f1, i)

        # Validation Phrase
        dev_loss, dev_ed_f1, dev_ae_f1 = run_over_data(data_iter=dev_iter,
                                                      optimizer=optimizer,
                                                      model=model,
                                                      need_backward=False,
                                                      tester=tester,
                                                      device=model.device,
                                                      maxnorm=parser.maxnorm,
                                                      label_i2s=parser.label_i2s,
                                                      save_output=False,
                                                      word_count_emb=parser.word_count_emb,
                                                      word_num_emb=parser.word_num_emb
                                                       )
        print("\nEpoch", i + 1, " dev loss: ", dev_loss,
              "\ndev ed f1: ", dev_ed_f1,
              " dev ae f1: ", dev_ae_f1)
        parser.writer.add_scalar('dev/loss', dev_loss, i)
        parser.writer.add_scalar('dev/ed/f1', dev_ed_f1, i)
        parser.writer.add_scalar('dev/ae/f1', dev_ae_f1, i)

        start = time.time()
        # Testing Phrase
        test_loss, test_ed_f1, test_ae_f1 = run_over_data(data_iter=test_iter,
                                                         optimizer=optimizer,
                                                         model=model,
                                                         need_backward=False,
                                                         tester=tester,
                                                         device=model.device,
                                                         maxnorm=parser.maxnorm,
                                                         label_i2s=parser.label_i2s,
                                                         save_output=os.path.join(parser.out, "test_output.csv"),
                                                         word_count_emb=parser.word_count_emb,
                                                         word_num_emb=parser.word_num_emb
                                                          )
        print('test an epoch:', time.time() - start)
        print("\nEpoch", i + 1, " test loss: ", test_loss,
              "\ntest ed f1: ", test_ed_f1,
              " test ae f1: ", test_ae_f1)
        
        parser.writer.add_scalar('test/loss', test_loss, i)
        parser.writer.add_scalar('test/ed/f1', test_ed_f1, i)
        parser.writer.add_scalar('test/ae/f1', test_ae_f1, i)

        # Early Stop
        aim = dev_ed_f1
        if scores <= aim:
            scores = aim
            # Move model parameters to CPU
            model.save_model(os.path.join(parser.out, "model.pt"))
            print("Save CPU model at Epoch", i + 1, "score: ", scores)
            print("Now testset score: ", test_ed_f1)
            now_bad = 0
            best_scores = test_ed_f1
            best_epoch = i + 1
        else:
            now_bad += 1
            if now_bad >= parser.earlystop:
                if restart_used >= parser.restart:
                    print("Restart opportunity are run out")
                    break
                restart_used += 1
                # lr = lr * 0.7
                model.load_model(os.path.join(parser.out, "model.pt"))
                lr = lr * 0.7
                optimizer = optimizer_constructor(lr=lr)
                print("Restart in Epoch %d" % (i + 2))
                now_bad = 0

        if test_ed_f1 > best_test_scores:
            best_test_scores = test_ed_f1
            # save best model parameters
            model.save_model(os.path.join(parser.out, "best_model.pt"))
            print("Save best model at Epoch", i + 1, "best_score: ", best_test_scores)

    print("Training Done!")
    print("Best score:", best_scores, "at epoch", best_epoch)
